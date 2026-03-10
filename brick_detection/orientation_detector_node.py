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
            'collab_ws', 'src', 'detection_grasping','brick_detection','weights', 'last.pt'
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

    def get_orientation_min_area(self, poly_points, brick_type, binary_mask):
        if len(poly_points) < 3 or binary_mask is None:
            return 0.0, None, (0.0, 0.0)
        
        # 1. Get the basic geometry
        pts = np.array(poly_points, dtype=np.float32).reshape(-1, 1, 2)
        rect = cv2.minAreaRect(pts)
        (rect_cx, rect_cy), (w, h), angle = rect
        

        # --- I-BRICK SPECIFIC LOGIC ---
        if brick_type == Brick.I_BRICK:
            # Determine which side is the long side
            if w < h:
                # The 'h' is the length, angle is usually negative from vertical
                actual_angle = angle + 90 
            else:
                # The 'w' is the length, angle is from horizontal
                actual_angle = angle + 180
                
            angle_rad = math.radians(actual_angle)
            
            # Symmetrize: For I-bricks, 0 deg and 180 deg are identical.
            # We force the arrow to always point in the 'positive' half-plane
            # to prevent 180-degree jitter.
            angle_rad = math.atan2(math.sin(angle_rad), math.cos(angle_rad))
            # if angle_rad > math.pi / 2:
            #     angle_rad -= math.pi
            # elif angle_rad < -math.pi / 2:
            #     angle_rad += math.pi
                
            box_points = cv2.boxPoints(rect).astype(int)
            return angle_rad, box_points, (rect_cx, rect_cy)
        # Standardize the angle based on long/short side
        if w < h:
            angle_rad = math.radians(angle + 180) # Vertical-ish
        else:
            angle_rad = math.radians(angle + 90)  # Horizontal-ish
        
       
        # Ensure angle is in [-pi, pi]
        angle_rad = math.atan2(math.sin(angle_rad), math.cos(angle_rad))

        # I-Bricks are symmetric; we don't care about 180-degree flips
        # if brick_type == Brick.I_BRICK:
        #     box_points = cv2.boxPoints(rect).astype(int)
        #     angle_rad = math.atan2(math.sin(math.radians(angle + 90)), math.cos(math.radians(angle_rad +90)))
        #     return angle_rad, box_points, (rect_cx, rect_cy)

        
        # 2. Sampling Strategy for L and T bricks
        # We check pixel density at +/- 'offset' along the calculated axis
        h_img, w_img = binary_mask.shape
        offset = 20  # Pixels to look ahead/behind
        
        def get_mask_density(center_x, center_y, ang, dist):
            # Calculate a point 'dist' away from center at angle 'ang'
            test_x = int(center_x + dist * math.cos(ang))
            test_y = int(center_y + dist * math.sin(ang))
            
            # Create a small 5x5 window to average density (more robust than 1 pixel)
            x_start, x_end = max(0, test_x-2), min(w_img, test_x+2)
            y_start, y_end = max(0, test_y-2), min(h_img, test_y+2)
            
            region = binary_mask[y_start:y_end, x_start:x_end]
            return np.sum(region) if region.size > 0 else 0

        # Sample both ends
        density_front = get_mask_density(rect_cx, rect_cy, angle_rad, offset)
        density_back = get_mask_density(rect_cx, rect_cy, angle_rad, -offset)
        def get_mask_density(cx, cy, ang, dist, mask):
            """
            Calculates the average pixel intensity in a small region 
            at a specific distance and angle from a center point.
            """
            h_img, w_img = mask.shape
            
            # 1. Calculate the coordinates of the 'probe' point
            # Using polar-to-cartesian conversion: x = r*cos(theta), y = r*sin(theta)
            tx = int(cx + dist * math.cos(ang))
            ty = int(cy + dist * math.sin(ang))
            
            # 2. Define a small 5x5 window (kernel) around that point
            # We use max/min to ensure we don't look outside the image boundaries
            x1, x2 = max(0, tx - 2), min(w_img, tx + 2)
            y1, y2 = max(0, ty - 2), min(h_img, ty + 2)
            
            # 3. Extract the region from the binary mask
            region = mask[y1:y2, x1:x2]
            
            # 4. Return the density (0.0 to 1.0)
            # np.mean on a binary mask (0s and 1s) returns the percentage of 'filled' pixels
            return np.mean(region) if region.size > 0 else 0
        if brick_type == Brick.L_BRICK:
            if w < h:
                base_angle = math.radians(angle + 180)
            else:
                base_angle = math.radians(angle + 90)

            corner_candidates = [
                base_angle + math.pi/4,    # 45 deg
                base_angle + 3*math.pi/4,  # 135 deg
                base_angle + 5*math.pi/4,  # 225 deg
                base_angle + 7*math.pi/4   # 315 deg
            ]

            # Scoring: We want the corner with the LEAST total plastic 
            # across two different radii (Inner and Outer)
            best_score = 999 # Higher than max possible (2.0)
            best_angle = base_angle
            self.debug_pts = [] # Store points for visualization
            for cand_angle in corner_candidates:
                # Inner check: 15 pixels (Near the center, should be plastic if not the gap)
                # Outer check: 28 pixels (In the middle of the 1x1 block area)
                

                r_inner, r_outer = 15, 28
                
                d1 = get_mask_density(rect_cx, rect_cy, cand_angle, r_inner, binary_mask)
                d2 = get_mask_density(rect_cx, rect_cy, cand_angle, r_outer, binary_mask)
                total_density = d1 + d2
                # Store points for drawing (x, y, density_value)
                p1 = (int(rect_cx + r_inner * math.cos(cand_angle)), int(rect_cy + r_inner * math.sin(cand_angle)))
                p2 = (int(rect_cx + r_outer * math.cos(cand_angle)), int(rect_cy + r_outer * math.sin(cand_angle)))
                self.debug_pts.append((p1, d1))
                self.debug_pts.append((p2, d2))

                
                if total_density < best_score:
                    best_score = total_density
                    best_angle = cand_angle
            
            angle_rad = math.atan2(math.sin(best_angle), math.cos(best_angle))
                
        elif brick_type == Brick.T_BRICK:
            # T-Brick: Point arrow toward the TAIL
            # The tail usually has higher density at the end of the long axis 
            # (depending on how your model defines the rectangle)
            if density_back > density_front:
                angle_rad += math.pi

        # Clean up angle and get box
        angle_rad = math.atan2(math.sin(angle_rad), math.cos(angle_rad))
        box_points = cv2.boxPoints(rect).astype(int)
        
        return angle_rad, box_points, (rect_cx, rect_cy)
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
        split_y = int(0.42 * H)
        
        grid_size_cm = 24.0
        grid_size_px = int(grid_size_cm * self.px_per_cm)
        
        grid_center_x = int((W / 2) + 56) 
        grid_center_y = split_y
        
        grid_x1 = int(grid_center_x - grid_size_px / 2)
        grid_y1 = int(grid_center_y - grid_size_px / 2)
        grid_x2 = int(grid_center_x + grid_size_px / 2)
        grid_y2 = int(grid_center_y + grid_size_px / 2)

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

            ### Uncomment for debugging only
            # self.get_logger().info(f"name:{name}")
            # if str(name) == "L":
            #     for pt, density in self.debug_pts:
            #         # Color logic: Green if it detects plastic (high density), Red if empty (low density)
            #         color = (0, 255, 0) if density > 0.5 else (0, 0, 255)
            #         cv2.circle(annotated_frame, pt, 3, color, -1)
                    
            #         # Optional: Draw faint circles showing the scan rings
            #         cv2.circle(annotated_frame, (int(centroid_x), int(centroid_y)), 15, (100, 100, 100), 1)
            #         cv2.circle(annotated_frame, (int(centroid_x), int(centroid_y)), 28, (100, 100, 100), 1)

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

            assigned_side = 0
            side_str = ""
            in_grid = (grid_x1 < cx < grid_x2) and (grid_y1 < cy < grid_y2)
            if in_grid:
                assigned_side = Brick.GRID
                side_str = "GRID"
            elif cy < split_y:
                assigned_side = Brick.ABB
                side_str = "ABB"
            else:
                assigned_side = Brick.AR4
                side_str = "AR4"
            # 5. Build Brick Message & Markers
            brick = Brick()
            brick.id = int(brick_id)
            brick.type = self.get_brick_type_id(name)
            brick.pose.position.x = (cx - self.intrinsics['cx']) * self.static_z / self.intrinsics['fx']
            brick.pose.position.y = (cy - self.intrinsics['cy']) * self.static_z / self.intrinsics['fy']
            brick.pose.position.z = float(self.static_z)
            brick.side = assigned_side
            # angle_rad -= math.pi/ 2
            # self.get_logger().info(f"Angle Degree: {math.degrees(angle_rad)}")
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
        
        overlay = annotated_frame.copy()
        cv2.line(overlay, (0, split_y), (W, split_y), (255, 255, 255), 2)
        cv2.putText(overlay, "ABB Side", (10, split_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, "AR4 Side", (10, split_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.rectangle(overlay, (grid_x1, grid_y1), (grid_x2, grid_y2), (0, 255, 255), 2)
        cv2.putText(overlay, "GRID", (grid_x1, grid_y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
        
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