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

from dual_arms_msgs.msg import BricksArray, Brick
from brick_detection.brick_tracker import BrickTracker

class YoloV8Detector(Node):
    def __init__(self):
        super().__init__('yolov8_detector')

        # --- Parameters ---
        default_model_path = os.path.join(
            os.path.expanduser('~'),
            'gp_ws', 'src', 'detection_grasping', 'brick_detection', 'weights', 'last.pt'
        )
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('image_topic', '/environment_camera/image_raw')
        self.declare_parameter('pixels_per_cm', 8.0) 

        model_path = self.get_parameter('model_path').value
        image_topic = self.get_parameter('image_topic').value
        self.px_per_cm = self.get_parameter('pixels_per_cm').value

        self.get_logger().info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        
        self.tracker = BrickTracker(distance_threshold=60, max_disappeared=300)
        self.bridge = CvBridge()
        
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.image_pub = self.create_publisher(Image, '/yolo/annotated_image', 10)
        self.dets_pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)
        self.bricks_pub = self.create_publisher(BricksArray, '/bricks_detected', 10)

    def get_quaternion_from_yaw(self, yaw):
        q = Quaternion()
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

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            return

        H, W, _ = frame.shape
        split_y = int(0.40 * H)
        
        # Grid Definition
        grid_size_px = int(24.0 * self.px_per_cm)
        grid_x1 = int(W/2 - grid_size_px/2)
        grid_y1 = int(split_y - grid_size_px/2)
        grid_x2 = int(W/2 + grid_size_px/2)
        grid_y2 = int(split_y + grid_size_px/2)

        # 2. Run YOLO
        results = self.model(frame, verbose=False, retina_masks=True)[0]
        current_frame_data = []

        if results.boxes is not None:
            for i, box in enumerate(results.boxes):
                # Basic Box info (Fallback)
                x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]

                # --- NEW LOGIC: Use Mask for Geometry ---
                cx, cy = (x1 + x2)/2, (y1 + y2)/2
                angle_deg = 0.0
                width_r = (x2 - x1)
                height_r = (y2 - y1)
                
                if results.masks is not None:
                    # Get the mask contour points
                    # mask.xy is a list of [N, 2] arrays of (x,y) points
                    contour = results.masks.xy[i].astype(np.int32)
                    
                    if len(contour) >= 3:
                        # --- THIS IS THE MAGIC FIX ---
                        # Use minAreaRect to get the REAL rotated box
                        rect = cv2.minAreaRect(contour)
                        (cx, cy), (w_box, h_box), angle_deg = rect
                        
                        # minAreaRect angle is tricky (-90 to 0 or 0 to 90 depending on version)
                        # We normalize width to always be the longer side for consistency
                        if w_box < h_box:
                            width_r, height_r = h_box, w_box
                            angle_deg += 90.0
                        else:
                            width_r, height_r = w_box, h_box
                
                # Convert angle to radians for ROS
                angle_rad = np.radians(angle_deg)

                detection_entry = {
                    'center': (cx, cy),
                    'type': class_name,
                    'box': (x1, y1, x2, y2), # Keep AABB for tracking association
                    'rotated_dim': (width_r, height_r),
                    'conf': conf,
                    'angle': angle_rad,
                    'id': None 
                }
                current_frame_data.append(detection_entry)

        # 3. Update Tracker
        tracked_detections = self.tracker.update(current_frame_data)

        # 4. Messages
        dets_msg = Detection2DArray(); dets_msg.header = msg.header
        bricks_msg = BricksArray(); bricks_msg.header = msg.header
        annotated_frame = frame.copy()

        for det in tracked_detections:
            brick_id = det['id']
            name = det['type']
            angle_rad = det['angle']
            cx, cy = det['center']
            w_rot, h_rot = det['rotated_dim']
            
            # Draw Standard Box for reference
            x1_box, y1_box, x2_box, y2_box = map(int, det['box'])
            
            # Determine Side
            in_grid = (grid_x1 < cx < grid_x2) and (grid_y1 < cy < grid_y2)
            side = Brick.GRID if in_grid else (Brick.ABB if cy < split_y else Brick.AR4)
            side_str = "GRID" if in_grid else ("ABB" if cy < split_y else "AR4")

            # --- Fill Brick Msg ---
            brick = Brick()
            brick.header = msg.header
            brick.id = int(brick_id)
            brick.type = self.get_brick_type_id(name)
            brick.side = side
            brick.pose.position.x = float(cx - W/2) 
            brick.pose.position.y = float(cy - H/2)
            brick.pose.orientation = self.get_quaternion_from_yaw(angle_rad)
            bricks_msg.bricks.append(brick)

            # --- Fill Detection2D Msg (Corrected with Rotated Size) ---
            ros_det = Detection2D()
            ros_det.header = msg.header
            ros_det.id = str(brick_id)
            ros_det.bbox.center.position.x = cx
            ros_det.bbox.center.position.y = cy
            ros_det.bbox.center.theta = angle_rad 
            
            # HERE IS THE FIX: Using the minAreaRect dimensions
            ros_det.bbox.size_x = float(w_rot)
            ros_det.bbox.size_y = float(h_rot)
            
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = name
            hyp.hypothesis.score = det['conf']
            ros_det.results.append(hyp)
            dets_msg.detections.append(ros_det)

            # --- Visualization ---
            color = (0, 255, 0)
            if side == Brick.GRID: color = (0, 255, 255)
            elif side == Brick.ABB: color = (255, 0, 0)
            
            # Draw Rotated Box (The "Real" Box)
            rect_struct = ((cx, cy), (w_rot, h_rot), np.degrees(angle_rad))
            box_pts = cv2.boxPoints(rect_struct)
            box_pts = np.int0(box_pts)
            cv2.drawContours(annotated_frame, [box_pts], 0, color, 2)
            
            label = f"ID:{brick_id} {name} {side_str}"
            cv2.putText(annotated_frame, label, (int(cx)-20, int(cy)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 7. Publish
        self.dets_pub.publish(dets_msg)
        self.bricks_pub.publish(bricks_msg)
        
        out_img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
        out_img_msg.header = msg.header
        self.image_pub.publish(out_img_msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(YoloV8Detector())
    rclpy.shutdown()

if __name__ == '__main__':
    main()