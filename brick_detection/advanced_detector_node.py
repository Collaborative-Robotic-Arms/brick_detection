import os
import math
import numpy as np
import rclpy
from rclpy.node import Node
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Quaternion
from ultralytics import YOLO

# Import your custom messages
from dual_arms_msgs.msg import BricksArray, Brick
from brick_detection.brick_tracker import BrickTracker

# ==========================================
#  The Main ROS Node
# ==========================================
class YoloV8Detector(Node):
    def __init__(self):
        super().__init__('yolov8_detector')

        # --- Parameters ---
        default_model_path = os.path.join(
            os.path.expanduser('~'),
            'gp_ws', 'src', 'detection_grasping','brick_detection','weights', 'last.pt'
        )
        self.declare_parameter('model_path', default_model_path)
        # self.declare_parameter('image_topic', '/environment_camera/image_raw')
        
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        
        # Scale factor to convert "30cm" to pixels. 
        # You MUST tune this (e.g., if image width is 640px and covers 80cm real world, px_per_cm = 8)
        self.declare_parameter('pixels_per_cm', 8.0) 

        model_path = self.get_parameter('model_path').value
        image_topic = self.get_parameter('image_topic').value
        self.px_per_cm = self.get_parameter('pixels_per_cm').value

        self.get_logger().info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Tracker settings
        self.tracker = BrickTracker(distance_threshold=60, max_disappeared=300)

        self.bridge = CvBridge()
        
        # --- Publishers / Subscribers ---
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.image_pub = self.create_publisher(Image, '/yolo/annotated_image', 10)
        self.dets_pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)
        self.bricks_pub = self.create_publisher(BricksArray, '/bricks_detected', 10)

    def get_orientation_pca(self, contour_points):
        if len(contour_points) < 3: 
            return 0.0
        pts = np.array(contour_points, dtype=np.float64).reshape(-1, 2)
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
        vx, vy = eigenvectors[0][0], eigenvectors[0][1]
        angle_rad = math.atan2(vy, vx)
        return angle_rad

    def get_quaternion_from_yaw(self, yaw):
        """Converts a yaw angle (radians) to a geometry_msgs/Quaternion."""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    def get_brick_type_id(self, class_name):
        """Maps YOLO string class to Brick msg constant."""
        cn = class_name.upper()
        if 'I' in cn: return Brick.I_BRICK
        if 'L' in cn: return Brick.L_BRICK
        if 'T' in cn: return Brick.T_BRICK
        if 'Z' in cn: return Brick.Z_BRICK
        return 255 # Unknown

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return

        # 1. Prepare Geometry
        H, W, _ = frame.shape
        
        # --- REGION DEFINITION (Horizontal Split) ---
        # Split line at 40% of HEIGHT
        split_y = int(0.40 * H)
        
        # Grid Area: 30x30cm square
        # Center: X = Middle of image, Y = On the split line
        grid_size_cm = 24.0
        grid_size_px = int(grid_size_cm * self.px_per_cm)
        
        grid_center_x = int(W / 2)
        grid_center_y = split_y  # Centered on the dividing line
        
        grid_x1 = int(grid_center_x - grid_size_px / 2)
        grid_y1 = int(grid_center_y - grid_size_px / 2)
        grid_x2 = int(grid_center_x + grid_size_px / 2)
        grid_y2 = int(grid_center_y + grid_size_px / 2)

        # 2. Run YOLO Inference
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

                # Orientation
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

        # 3. Update Tracker
        tracked_detections = self.tracker.update(current_frame_data)

        # 4. Prepare Messages
        dets_msg = Detection2DArray()
        dets_msg.header = msg.header
        bricks_msg = BricksArray()
        bricks_msg.header = msg.header

        annotated_frame = frame.copy()

        # 5. Process Detections
        for det in tracked_detections:
            brick_id = det['id']
            name = det['type']
            angle_rad = det['angle']
            cx, cy = det['center']
            x1_box, y1_box, x2_box, y2_box = map(int, det['box'])

            # L-Shape Offset
            if 'L' in name.upper(): 
                 angle_rad -= (math.pi / 4)

            # --- DETERMINE SIDE (Horizontal Logic) ---
            # 1. Is it in Grid?
            in_grid = (grid_x1 < cx < grid_x2) and (grid_y1 < cy < grid_y2)
            
            assigned_side = 0
            side_str = ""
            
            if in_grid:
                assigned_side = Brick.GRID
                side_str = "GRID"
            elif cy < split_y:
                # Top 40%
                assigned_side = Brick.ABB
                side_str = "ABB"
            else:
                # Bottom 60%
                assigned_side = Brick.AR4
                side_str = "AR4"

            # --- Fill Brick Msg ---
            brick = Brick()
            brick.header = msg.header
            brick.id = int(brick_id)
            brick.type = self.get_brick_type_id(name)
            brick.side = assigned_side
            
            # Pose (Relative to Image Center)
            brick.pose.position.x = float(cx - W/2) 
            brick.pose.position.y = float(cy - H/2)
            brick.pose.position.z = 0.0
            brick.pose.orientation = self.get_quaternion_from_yaw(angle_rad)
            
            bricks_msg.bricks.append(brick)

            # --- Fill Detection2D Msg ---
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

            # --- Visualization ---
            color = (0, 255, 0)
            if side_str == "GRID": color = (0, 255, 255) # Yellow
            elif side_str == "ABB": color = (255, 0, 0)  # Blue (Top)
            elif side_str == "AR4": color = (0, 255, 0)  # Green (Bottom)
            
            cv2.rectangle(annotated_frame, (x1_box, y1_box), (x2_box, y2_box), color, 2)
            label = f"ID:{brick_id} {name} [{side_str}]"
            cv2.putText(annotated_frame, label, (x1_box, y1_box - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Orientation Axis
            axis_len = 40
            end_x = int(cx + axis_len * math.cos(angle_rad))
            end_y = int(cy + axis_len * math.sin(angle_rad))
            cv2.line(annotated_frame, (int(cx), int(cy)), (end_x, end_y), (0, 0, 255), 3)

        # 6. Draw Segmentation Visualization (Overlay)
        overlay = annotated_frame.copy()
        
        # Horizontal Split Line
        cv2.line(overlay, (0, split_y), (W, split_y), (255, 255, 255), 2)
        
        # Labels
        # Top Region
        cv2.putText(overlay, "ABB Side", (10, split_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Bottom Region
        cv2.putText(overlay, "AR4 Side", (10, split_y + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Grid Box
        cv2.rectangle(overlay, (grid_x1, grid_y1), (grid_x2, grid_y2), (0, 255, 255), 2)
        cv2.putText(overlay, "GRID", (grid_x1, grid_y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Blend
        cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)

        # 7. Publish
        self.dets_pub.publish(dets_msg)
        self.bricks_pub.publish(bricks_msg)
        
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