import os

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO
import cv2
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

class YoloV8Detector(Node):
    def __init__(self):
        super().__init__('yolov8_detector')

        # ===== Parameters =====
        default_model_path = os.path.join(
            os.path.expanduser('~'),
            'collab_arms_ws', 'src', 'brick_detection', 'weights', 'yolo_bricks_seg_best.pt'
        )
        # مسار الموديل (تقدر تغيره من الـ CLI)
        self.declare_parameter('model_path', default_model_path)

        # topic بتاع صورة الـ RealSense (اللي شوفناه في ros2 topic list)
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')

        model_path = self.get_parameter('model_path').value
        image_topic = self.get_parameter('image_topic').value

        self.get_logger().info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)

        self.bridge = CvBridge()

        # Subscriber: camera_image
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        # Publisher: bounding boxes on the bricks
        self.image_pub = self.create_publisher(
            Image,
            '/yolo/annotated_image',
            10
        )
        # Publisher للداتا الرقمية بتاعة الـ detections
        self.dets_pub = self.create_publisher(
            Detection2DArray,
            '/yolo/detections',
            10
        )
           


        self.get_logger().info(f"Subscribed to image topic: {image_topic}")
        self.get_logger().info("YOLO detector node started.")

    def image_callback(self, msg: Image):
        # ROS Image → OpenCV
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CV bridge error: {e}")
            return

        # شغّل YOLO
        results = self.model(frame, verbose=False)[0]

        # annotated image (boxes + masks)
        annotated = results.plot()

        # ابعت الصورة
        out_img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        out_img_msg.header = msg.header
        self.image_pub.publish(out_img_msg)

        # ====== Build Detection2DArray ======
        dets_msg = Detection2DArray()
        dets_msg.header = msg.header

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(float, [x1, y1, x2, y2])

                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]

                det = Detection2D()
                det.header = msg.header

                # center + size
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = x2 - x1
                h = y2 - y1

                det.bbox.center.position.x = cx
                det.bbox.center.position.y = cy
                det.bbox.size_x = w
                det.bbox.size_y = h

                # hypothesis (class + score)
                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = class_name
                hyp.hypothesis.score = conf
                det.results.append(hyp)

                dets_msg.detections.append(det)

        self.dets_pub.publish(dets_msg)

        if len(dets_msg.detections) > 0:
            self.get_logger().info(f"Published {len(dets_msg.detections)} detections.")


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

