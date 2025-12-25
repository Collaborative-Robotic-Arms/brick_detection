import os
import math
import numpy as np
import rclpy
from rclpy.node import Node
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics import YOLO

# ==========================================
# 1. The Tracker Class (Handles ID & Smoothing)
# ==========================================
class BrickTracker:
    def __init__(self, distance_threshold=60, max_disappeared=10):
        # Store state: {id: {'center': (x,y), 'type': 'L', 'disappeared': 0, 'smooth_vec': (vx, vy)}}
        self.tracked_objects = {}
        self.next_object_id = 0
        self.distance_threshold = distance_threshold
        self.max_disappeared = max_disappeared
        
        # Smoothing factor (0.0 = no new data, 1.0 = no smoothing). 
        # 0.2 means we trust the history 80% and new data 20%.
        self.alpha = 0.2 

    def update(self, detections):
        """
        detections: list of dicts.
        Returns: The list with 'id' and 'stable_angle' added.
        """
        # 1. Mark existing as potentially missing
        for obj_id in self.tracked_objects:
            self.tracked_objects[obj_id]['disappeared'] += 1

        # 2. Match Detections to IDs
        if len(detections) > 0:
            for det in detections:
                cx, cy = det['center']
                best_id = -1
                min_dist = self.distance_threshold

                for obj_id, data in self.tracked_objects.items():
                    # Strict type matching
                    if data['type'] != det['type']:
                        continue

                    dist = math.hypot(cx - data['center'][0], cy - data['center'][1])
                    if dist < min_dist:
                        min_dist = dist
                        best_id = obj_id

                # 3. Update State
                if best_id != -1:
                    # Found match
                    self.tracked_objects[best_id]['center'] = (cx, cy)
                    self.tracked_objects[best_id]['disappeared'] = 0
                    
                    # --- STABILIZE ANGLE ---
                    raw_angle = det['angle']
                    stable_angle = self.smooth_angle(best_id, raw_angle)
                    
                    det['id'] = best_id
                    det['angle'] = stable_angle # Overwrite with smooth angle
                else:
                    # New object
                    new_id = self.register(det)
                    det['id'] = new_id
                    # Initial angle is just the raw angle
                    det['angle'] = det['angle'] 

        # 4. Cleanup
        self.cleanup()
        return detections

    def smooth_angle(self, obj_id, raw_angle):
        """
        Smooths angle using vector averaging to handle the 360-wrap 
        and PCA 180-flip ambiguity.
        """
        # Get previous smoothed vector (vx, vy)
        prev_vx, prev_vy = self.tracked_objects[obj_id]['smooth_vec']
        
        # Convert new raw angle to vector
        new_vx = math.cos(raw_angle)
        new_vy = math.sin(raw_angle)

        # --- Fix 180-degree Flip (PCA Ambiguity) ---
        # If dot product is negative, the vectors point in opposite directions.
        # Flip the new one to match the "flow" of the previous one.
        dot_product = (prev_vx * new_vx) + (prev_vy * new_vy)
        if dot_product < 0:
            new_vx = -new_vx
            new_vy = -new_vy

        # --- Exponential Moving Average (EMA) ---
        # avg = (1-alpha)*old + alpha*new
        avg_vx = (1 - self.alpha) * prev_vx + self.alpha * new_vx
        avg_vy = (1 - self.alpha) * prev_vy + self.alpha * new_vy

        # Update stored vector
        self.tracked_objects[obj_id]['smooth_vec'] = (avg_vx, avg_vy)
        
        # Convert back to radians
        return math.atan2(avg_vy, avg_vx)

    def register(self, det):
        obj_id = self.next_object_id
        # Initialize smooth vector from the very first detection
        vx = math.cos(det['angle'])
        vy = math.sin(det['angle'])
        
        self.tracked_objects[obj_id] = {
            'center': det['center'],
            'type': det['type'],
            'disappeared': 0,
            'smooth_vec': (vx, vy) 
        }
        self.next_object_id += 1
        return obj_id

    def cleanup(self):
        ids_to_delete = [
            obj_id for obj_id, data in self.tracked_objects.items()
            if data['disappeared'] > self.max_disappeared
        ]
        for obj_id in ids_to_delete:
            del self.tracked_objects[obj_id]

# ==========================================
# 2. The Main ROS Node
# ==========================================
class YoloV8Detector(Node):
    def __init__(self):
        super().__init__('yolov8_detector')

        # Use an expanded path or relative path as needed
        default_model_path = os.path.join(
            os.path.expanduser('~'),
            'gp_ws', 'src', 'brick_detection', 'weights', 'last.pt'
        )
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')

        model_path = self.get_parameter('model_path').value
        image_topic = self.get_parameter('image_topic').value

        self.get_logger().info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        
        # Increased thresholds for stability
        self.tracker = BrickTracker(distance_threshold=60, max_disappeared=10)

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.image_pub = self.create_publisher(Image, '/yolo/annotated_image', 10)
        self.dets_pub = self.create_publisher(Detection2DArray, '/yolo/detections', 10)

    def get_orientation_pca(self, contour_points):
        if len(contour_points) < 3: 
            return 0.0
        pts = np.array(contour_points, dtype=np.float64).reshape(-1, 2)
        mean, eigenvectors, eigenvalues = cv2.PCACompute2(pts, mean=None)
        vx, vy = eigenvectors[0][0], eigenvectors[0][1]
        angle_rad = math.atan2(vy, vx)
        return angle_rad

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return

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

                # Get RAW PCA orientation
                orientation = 0.0
                if has_masks:
                    poly = results.masks.xy[i]
                    orientation = self.get_orientation_pca(poly)

                detection_entry = {
                    'center': (cx, cy),
                    'type': class_name,
                    'box': (x1, y1, x2, y2),
                    'conf': conf,
                    'angle': orientation, # Raw angle goes in
                    'id': None 
                }
                current_frame_data.append(detection_entry)

        # ====== Update IDs & Stabilize Angles ======
        # The tracker now modifies 'angle' inside the dict to be the stable one
        tracked_detections = self.tracker.update(current_frame_data)

        # ====== Build ROS Message ======
        dets_msg = Detection2DArray()
        dets_msg.header = msg.header
        annotated_frame = frame.copy()

        for det in tracked_detections:
            brick_id = det['id']
            name = det['type']
            angle_rad = det['angle'] # This is now the SMOOTHED angle from tracker
            
            # --- CUSTOM LOGIC: L-SHAPE OFFSET ---
            # You requested adding 45 degrees (pi/4) if it's an L shape.
            # We do this AFTER smoothing so the smoother doesn't fight the offset.
            if name == 'L_shape' or name == 'L': 
                 angle_rad -= (math.pi / 4)

            # Draw
            x1, y1, x2, y2 = map(int, det['box'])
            cx, cy = det['center']
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{brick_id} {name}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Axis line
            axis_len = 50
            end_x = int(cx + axis_len * math.cos(angle_rad))
            end_y = int(cy + axis_len * math.sin(angle_rad))
            cv2.line(annotated_frame, (int(cx), int(cy)), (end_x, end_y), (0, 0, 255), 3)

            end_x_perp = int(cx + axis_len * math.cos(angle_rad+math.pi/2))
            end_y_perp = int(cy + axis_len * math.sin(angle_rad+math.pi/2))
            cv2.line(annotated_frame, (int(cx), int(cy)), (end_x_perp, end_y_perp), (0, 255, 0), 3)
            # ROS Msg
            ros_det = Detection2D()
            ros_det.header = msg.header
            ros_det.id = str(brick_id)
            ros_det.bbox.center.position.x = cx
            ros_det.bbox.center.position.y = cy
            ros_det.bbox.center.theta = angle_rad # Final logic with offset
            ros_det.bbox.size_x = float(x2 - x1)
            ros_det.bbox.size_y = float(y2 - y1)

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = name
            hyp.hypothesis.score = det['conf']
            ros_det.results.append(hyp)
            dets_msg.detections.append(ros_det)

        self.dets_pub.publish(dets_msg)
        
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