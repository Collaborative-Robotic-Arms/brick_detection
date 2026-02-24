import os
import math
import numpy as np
import rclpy
from rclpy.node import Node
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Quaternion, Pose
from ultralytics import YOLO
from visualization_msgs.msg import Marker, MarkerArray

# Import your custom messages
try:
    from dual_arms_msgs.msg import BricksArray, Brick
    from dual_arms_msgs.srv import DetectBricks
except ImportError:
    pass

from brick_detection.brick_tracker import BrickTracker

class YoloV8Detector(Node):
    def __init__(self):
        super().__init__('yolov8_detector')

        # --- Parameters ---
        default_model_path = os.path.join(
            os.path.expanduser('~'),
            'gp_ws', 'src', 'detection_grasping','brick_detection','weights', 'last.pt'
        )
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('pixels_per_cm', 8.0) 
        self.declare_parameter('static_z_height', 0.712) 
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')

        model_path = self.get_parameter('model_path').value
        image_topic = self.get_parameter('image_topic').value
        self.px_per_cm = self.get_parameter('pixels_per_cm').value
        self.static_z = self.get_parameter('static_z_height').value
        self.camera_frame = self.get_parameter('camera_frame').value

        # --- Calibration Data ---
        self.k_matrix = np.array([[607.649, 0.0, 330.204], [0.0, 605.196, 246.368], [0.0, 0.0, 1.0]])
        self.dist_coeffs = np.array([0.0307, 0.6603, -0.0030, -0.0053, -2.5473])

        self.img_w, self.img_h = 640, 480 
        new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(self.k_matrix, self.dist_coeffs, (self.img_w, self.img_h), 0)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.k_matrix, self.dist_coeffs, None, new_camera_mtx, (self.img_w, self.img_h), cv2.CV_32FC1)

        self.intrinsics = {'fx': new_camera_mtx[0,0], 'fy': new_camera_mtx[1,1], 'cx': new_camera_mtx[0,2], 'cy': new_camera_mtx[1,2]}

        self.model = YOLO(model_path)
        self.tracker = BrickTracker(distance_threshold=60, max_disappeared=300)
        self.bridge = CvBridge()
        self.last_bricks_detected = BricksArray()
        
        # Publishers
        self.image_pub = self.create_publisher(Image, '/yolo/annotated_image', 10)
        self.dets_pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)
        self.bricks_pub = self.create_publisher(BricksArray, '/detected_bricks', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/yolo/markers', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.srv = self.create_service(DetectBricks, 'detect_bricks', self.detect_bricks_callback)

    def get_orientation_min_area(self, poly_points, brick_type, binary_mask=None):
        if len(poly_points) < 3:
            return 0.0, None, (0.0, 0.0)
        
        pts = np.array(poly_points, dtype=np.float32).reshape(-1, 1, 2)
        rect = cv2.minAreaRect(pts)
        (rect_cx, rect_cy), (w, h), angle = rect
        

        M = cv2.moments(pts)
        if M['m00'] == 0: return 0.0, None, (rect_cx, rect_cy)
        centroid_x, centroid_y = M['m10']/M['m00'], M['m01']/M['m00']
        
        vec_x, vec_y = centroid_x - rect_cx, centroid_y - rect_cy
        
        if brick_type == Brick.L_BRICK:
            angle_rad = math.atan2(vec_y, vec_x)
        elif brick_type == Brick.I_BRICK:
            angle_rad = math.radians(angle)
        else:
            if w < h: angle += 90
            angle_rad = math.radians(angle)

        if binary_mask is not None:
            check_dist = 15 
            tail_x = int(rect_cx - check_dist * math.cos(angle_rad))
            tail_y = int(rect_cy - check_dist * math.sin(angle_rad))
            h_img, w_img = binary_mask.shape
            if 0 <= tail_x < w_img and 0 <= tail_y < h_img:
                if binary_mask[tail_y, tail_x] == 0:
                    angle_rad += math.pi

        angle_rad = math.atan2(math.sin(angle_rad), math.cos(angle_rad))
        box_points = cv2.boxPoints(rect).astype(int)
        return angle_rad, box_points, (centroid_x, centroid_y)

    def get_quaternion_from_yaw(self, yaw):
        return Quaternion(x=0.0, y=0.0, z=math.sin(yaw/2.0), w=math.cos(yaw/2.0))

    def get_brick_type_id(self, class_name):
        cn = class_name.upper()
        if 'I' in cn: return Brick.I_BRICK
        if 'L' in cn: return Brick.L_BRICK
        if 'T' in cn: return Brick.T_BRICK
        return 255

    def image_callback(self, msg: Image):
        try:
            raw_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception: return

        frame = cv2.remap(raw_frame, self.map1, self.map2, cv2.INTER_LINEAR)
        H, W, _ = frame.shape
        
        results = self.model(frame, verbose=False, retina_masks=True)[0]
        current_frame_data = []

        if results.boxes is not None:
            has_masks = results.masks is not None
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                brick_type_id = self.get_brick_type_id(class_name)
                
                orientation, rotated_box_pts, centroid = 0.0, None, ((x1+x2)/2, (y1+y2)/2)
                if has_masks:
                    poly = results.masks.xy[i]
                    mask_data = results.masks.data[i].cpu().numpy().astype(np.uint8)
                    orientation, rotated_box_pts, centroid = self.get_orientation_min_area(poly, brick_type_id, mask_data)

                current_frame_data.append({
                    'center': ((x1+x2)/2, (y1+y2)/2),
                    'type': class_name,
                    'box': (x1, y1, x2, y2),
                    'conf': float(box.conf[0]),
                    'angle': orientation,
                    'rotated_box': rotated_box_pts,
                    'centroid': centroid,
                    'id': None 
                })

        tracked_detections = self.tracker.update(current_frame_data)
        
        bricks_msg = BricksArray()
        bricks_msg.header = msg.header
        bricks_msg.header.frame_id = self.camera_frame
        
        dets_msg = Detection2DArray()
        dets_msg.header = msg.header
        
        marker_array = MarkerArray()
        annotated_frame = frame.copy()
    
        for det in tracked_detections:
            cx, cy = det['center']
            centroid_x, centroid_y = det['centroid']
            angle_rad = det['angle']
            x1_box, y1_box, x2_box, y2_box = map(int, det['box'])
            brick_id = det['id']
            name = det['type']

            # 1. Draw Centroid (Magenta) and Offset line
            cv2.circle(annotated_frame, (int(centroid_x), int(centroid_y)), 5, (255, 0, 255), -1)
            cv2.line(annotated_frame, (int(cx), int(cy)), (int(centroid_x), int(centroid_y)), (200, 200, 200), 2)

            # 2. Draw Yellow Arrow (Heading)
            arrow_len = 50
            a_end_x = int(cx + arrow_len * math.cos(angle_rad))
            a_end_y = int(cy + arrow_len * math.sin(angle_rad))
            cv2.arrowedLine(annotated_frame, (int(cx), int(cy)), (a_end_x, a_end_y), (0, 255, 255), 3, tipLength=0.3)

            # 3. Draw Rotated Bounding Box (Blue)
            if det['rotated_box'] is not None:
                cv2.drawContours(annotated_frame, [det['rotated_box']], 0, (255, 0, 0), 2)

            # 4. Populate vision_msgs/Detection2D
            ros_det = Detection2D()
            ros_det.header = msg.header
            ros_det.id = str(brick_id)
            ros_det.bbox.center.position.x = cx
            ros_det.bbox.center.position.y = cy
            ros_det.bbox.center.theta = angle_rad
            ros_det.bbox.size_x = float(x2_box - x1_box)
            ros_det.bbox.size_y = float(y2_box - y1_box)
            
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = name
            hyp.hypothesis.score = det['conf']
            ros_det.results.append(hyp)
            dets_msg.detections.append(ros_det)

            # 5. Build Brick Message & Markers
            brick = Brick()
            brick.id = int(brick_id)
            brick.type = self.get_brick_type_id(name)
            brick.pose.position.x = (cx - self.intrinsics['cx']) * self.static_z / self.intrinsics['fx']
            brick.pose.position.y = (cy - self.intrinsics['cy']) * self.static_z / self.intrinsics['fy']
            brick.pose.position.z = float(self.static_z)

            # angle_rad -= math.pi/ 2
            self.get_logger().info(f"Angle Degree: {math.degrees(angle_rad)}")
            brick.pose.orientation = self.get_quaternion_from_yaw(angle_rad)
            bricks_msg.bricks.append(brick)

            marker = Marker()
            marker.header = bricks_msg.header
            marker.id = brick_id
            marker.type = Marker.CUBE
            marker.pose = brick.pose
            marker.scale.x, marker.scale.y, marker.scale.z = 0.03, 0.03, 0.03
            marker.color.r, marker.color.a = 0.8, 0.8
            marker_array.markers.append(marker)

        # Publish all
        self.dets_pub.publish(dets_msg)
        self.bricks_pub.publish(bricks_msg)
        self.marker_pub.publish(marker_array)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8'))

    def detect_bricks_callback(self, request, response):
        req_type_str = request.brick_type.upper() if request.brick_type else "ALL"
        matched = []
        for b in self.last_bricks_detected.bricks:
            if req_type_str == "ALL" or b.type == self.get_brick_type_id(req_type_str):
                matched.append(b)
        response.bricks = matched
        response.success = len(matched) > 0
        return response

def main(args=None):
    rclpy.init(args=args)
    node = YoloV8Detector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()