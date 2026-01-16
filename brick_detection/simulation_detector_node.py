import os
import math
import numpy as np
import rclpy
from rclpy.node import Node
import cv2

from sensor_msgs.msg import Image, CameraInfo  # Added CameraInfo
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Quaternion, Pose
from visualization_msgs.msg import Marker, MarkerArray # Added Markers
from ultralytics import YOLO

# Import your custom messages
from dual_arms_msgs.msg import BricksArray, Brick
from brick_detection.brick_tracker import BrickTracker

# ==========================================
#  Service Definition
# ==========================================
try:
    from dual_arms_msgs.srv import DetectBricks
except ImportError:
    from dual_arms_msgs.srv import DetectBricks


# ==========================================
#  The Main ROS Node
# ==========================================
class YoloV8Detector(Node):
    def __init__(self):
        super().__init__('yolov8_detector')

        # --- Parameters ---
        default_model_path = os.path.join(
            os.path.expanduser('~'),
            'gp_ws', 'src', 'detection_grasping','brick_detection','weights', 'best_final.pt'
        )
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('image_topic', '/environment_camera/image_raw')
        
        # We try to guess the camera_info topic from the image topic, or set explicitly
        self.declare_parameter('camera_info_topic', '/environment_camera/camera_info')
        
        # CRITICAL: CameraInfo gives focal length (px), but to get size (cm), 
        # we need the physical distance from camera to the table/bricks.
        self.declare_parameter('work_plane_distance_m', 0.779) 

        model_path = self.get_parameter('model_path').value
        image_topic = self.get_parameter('image_topic').value
        c_info_topic = self.get_parameter('camera_info_topic').value
        self.work_distance = self.get_parameter('work_plane_distance_m').value

        # Initialize defaults in case CameraInfo hasn't arrived yet
        self.px_per_cm = 8.0 
        self.fx = 1000.0 # Default focal length placeholder
        self.fy = 1000.0
        self.cx = 640.0
        self.cy = 360.0
        self.camera_frame_id = "camera_optical_frame" 

        self.get_logger().info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Tracker settings
        self.tracker = BrickTracker(distance_threshold=60, max_disappeared=300)

        self.bridge = CvBridge()
        
        self.last_bricks_detected = BricksArray()
        
        # --- Publishers / Subscribers ---
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        
        # NEW: Camera Info Subscriber
        self.info_sub = self.create_subscription(CameraInfo, c_info_topic, self.camera_info_callback, 10)
        
        self.image_pub = self.create_publisher(Image, '/yolo/annotated_image', 10)
        self.dets_pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)
        self.bricks_pub = self.create_publisher(BricksArray, '/detected_bricks', 10)
        
        # NEW: Marker Publisher
        self.marker_pub = self.create_publisher(MarkerArray, '/yolo/markers', 10)

        # Service
        self.srv = self.create_service(
            DetectBricks, 
            'detect_bricks', 
            self.detect_bricks_callback
        )
        self.get_logger().info("Service 'detect_bricks' is ready.")

    # ==========================================
    #  NEW: Camera Info Callback
    # ==========================================
    def camera_info_callback(self, msg: CameraInfo):
        """
        Updates pixels_per_cm dynamically based on camera intrinsics
        and the known work plane distance.
        """
        self.camera_frame_id = msg.header.frame_id
        
        # Intrinsic Matrix K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

        if self.work_distance > 0:
            # Formula: px_per_cm = focal_length_px / distance_cm
            self.px_per_cm = self.fx / (self.work_distance * 100.0)
            
            # Optional: Log once just to verify (uncomment if debugging)
            # self.get_logger().info(f"Updated px_per_cm: {self.px_per_cm:.2f} (fx={self.fx:.1f}, Z={self.work_distance}m)", throttle_duration_sec=5.0)

    # ==========================================
    #  Service Callback
    # ==========================================
    def detect_bricks_callback(self, request, response):
        self.get_logger().info(f"Service Request Received. Looking for type: '{request.brick_type}'")
        req_type_str = request.brick_type.upper() if request.brick_type else "ALL"
        matched_bricks = []

        if self.last_bricks_detected and self.last_bricks_detected.bricks:
            for b in self.last_bricks_detected.bricks:
                if req_type_str == "ALL" or req_type_str == "":
                    matched_bricks.append(b)
                else:
                    target_id = self.get_brick_type_id(req_type_str)
                    if b.type == target_id:
                        matched_bricks.append(b)

        response.bricks = matched_bricks
        if matched_bricks:
            response.success = True
        else:
            response.success = False
            response.handover_pose = Pose()

        return response

    # ... (Helper methods) ...
    def get_orientation_pca(self, contour_points):
        if len(contour_points) < 3: 
            return 0.0
        pts = np.array(contour_points, dtype=np.float64).reshape(-1, 2)
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
        vx, vy = eigenvectors[0][0], eigenvectors[0][1]
        angle_rad = math.atan2(vy, vx)
        return angle_rad

    def get_quaternion_from_yaw(self, yaw):
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def get_brick_type_id(self, class_name):
        cn = class_name.upper()
        if 'I' in cn: return Brick.I_BRICK
        if 'L' in cn: return Brick.L_BRICK
        if 'T' in cn: return Brick.T_BRICK
        if 'Z' in cn: return Brick.Z_BRICK
        return 255 

    # ==========================================
    #  Image Callback (Main Logic)
    # ==========================================
    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return

        H, W, _ = frame.shape
        split_y = int(0.42 * H)
        
        # Grid settings using dynamic px_per_cm
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
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                conf = float(box.conf[0])

                orientation = 0.0
                if has_masks:
                    poly = results.masks.xy[i]
                    orientation = self.get_orientation_pca(poly)

                detection_entry = {
                    'center': (cx, cy),
                    'type': class_name,
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'angle': orientation,
                    'id': None 
                }
                current_frame_data.append(detection_entry)

        tracked_detections = self.tracker.update(current_frame_data)

        dets_msg = Detection2DArray()
        dets_msg.header = msg.header
        bricks_msg = BricksArray()
        bricks_msg.header = msg.header
        
        # NEW: Initialize Marker Array
        marker_array = MarkerArray()

        annotated_frame = frame.copy()

        for det in tracked_detections:
            brick_id = det['id']
            name = det['type']
            angle_rad = det['angle']
            cx, cy = det['center']
            x1_box, y1_box, x2_box, y2_box = map(int, det['box'])

            if 'L' in name.upper(): 
                 angle_rad -= (math.pi / 4)

            # --- Logic for Side Assignment ---
            in_grid = (grid_x1 < cx < grid_x2) and (grid_y1 < cy < grid_y2)
            assigned_side = 0
            side_str = ""
            if in_grid:
                assigned_side = Brick.GRID
                side_str = "GRID"
            elif cy < split_y:
                assigned_side = Brick.ABB
                side_str = "ABB"
            else:
                assigned_side = Brick.AR4
                side_str = "AR4"

            # --- Fill Brick Message ---
            brick = Brick()
            brick.header = msg.header
            brick.id = int(brick_id)
            brick.type = self.get_brick_type_id(name)
            brick.side = assigned_side
            
            # NOTE: Your original logic returns positions in PIXELS centered on image
            brick.pose.position.x = float(cx - W/2) 
            brick.pose.position.y = float(cy - H/2)
            brick.pose.position.z = 0.0
            brick.pose.orientation = self.get_quaternion_from_yaw(angle_rad)
            bricks_msg.bricks.append(brick)

            # --- Fill Detection2D Message ---
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

            # --- NEW: Create Marker ---
            marker = Marker()
            marker.header.frame_id = 'camera'  #self.camera_frame_id # Use the frame from CameraInfo
            marker.header.stamp = msg.header.stamp
            marker.ns = "detected_bricks"
            marker.id = int(brick_id)
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # Convert Pixel (cx, cy) to Meter (X, Y) using Pinhole Camera Model
            # X = Z * (u - cx) / fx
            # Y = Z * (v - cy) / fy
            # Z is self.work_distance
            X = self.work_distance * (cx - self.cx) / self.fx
            Y = self.work_distance * (cy - self.cy) / self.fy
            Z = self.work_distance # Distance from camera
            # Ensure we use the Principal Point (self.cx, self.cy) from CameraInfo, not W/2
            marker.pose.position.x = X
            marker.pose.position.y = Y
            marker.pose.position.z = Z
            

            # marker.pose.position.x = self.work_distance * (cx - self.cx) / self.fx
            # marker.pose.position.y = self.work_distance * (cy - self.cy) / self.fy
            # marker.pose.position.z = self.work_distance # Distance from camera
            
            # Orientation (Same as detection, rotated around Z)
            marker.pose.orientation = self.get_quaternion_from_yaw(angle_rad)
            
            # Scale: 0.1x0.1x0.1 as requested
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            
            # Color
            marker.color.a = 0.8
            if side_str == "GRID": 
                marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 1.0 # Cyan
            elif side_str == "ABB": 
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0 # Red
            else: 
                marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0 # Green

            marker_array.markers.append(marker)
            # ---------------------------

            # Drawing on Image
            color = (0, 255, 0)
            if side_str == "GRID": color = (0, 255, 255) 
            elif side_str == "ABB": color = (255, 0, 0) 
            elif side_str == "AR4": color = (0, 255, 0) 
            
            cv2.rectangle(annotated_frame, (x1_box, y1_box), (x2_box, y2_box), color, 2)
            label = f"ID:{brick_id} {name} [{side_str}]"
            cv2.putText(annotated_frame, label, (x1_box, y1_box - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            axis_len = 40
            end_x = int(cx + axis_len * math.cos(angle_rad))
            end_y = int(cy + axis_len * math.sin(angle_rad))
            cv2.line(annotated_frame, (int(cx), int(cy)), (end_x, end_y), (0, 0, 255), 3)

        # Drawing Overlay
        overlay = annotated_frame.copy()
        cv2.line(overlay, (0, split_y), (W, split_y), (255, 255, 255), 2)
        cv2.putText(overlay, "ABB Side", (10, split_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, "AR4 Side", (10, split_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.rectangle(overlay, (grid_x1, grid_y1), (grid_x2, grid_y2), (0, 255, 255), 2)
        cv2.putText(overlay, "GRID", (grid_x1, grid_y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)

        # Publish all
        self.dets_pub.publish(dets_msg)
        self.last_bricks_detected = bricks_msg
        self.bricks_pub.publish(bricks_msg)
        self.marker_pub.publish(marker_array) # Publish Markers
        
        out_img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        out_img_msg.header = msg.header
        self.image_pub.publish(out_img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloV8Detector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()