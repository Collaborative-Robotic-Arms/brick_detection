import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped

from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import Mesh, MeshTriangle
from geometry_msgs.msg import Point

import tf2_ros
import tf2_geometry_msgs
import numpy as np
import trimesh
import os

from message_filters import Subscriber, ApproximateTimeSynchronizer

# YOLO class_id → mesh type
CLASS_TO_TYPE = {'I': 0, 'L': 1, 'T': 2}


class Real2SimBridge(Node):

    def __init__(self):
        super().__init__('real2sim_bridge')

        # ============================================
        # PARAMETERS
        # ============================================
        self.declare_parameter('target_frame', 'world')
        self.declare_parameter('mesh_base_path',
            '/home/tarek/collab_ws/src/dual_arms_packages/dual_arms/models')

        # ---------------------------------------------------------------
        # world → camera_link  (the missing TF link)
        #
        # POSITION (from sim /tf_static chain):
        #   world → abb_table : (0.3325, 0.0,  1.1)
        #   abb_table → camera: (0.670,  0.01, 0.8035)
        #   ────────────────────────────────────────
        #   world → camera_link: (1.0025, 0.01, 1.9035)
        #
        # ROTATION:
        #   Camera is physically TOP-DOWN (lens pointing at table = world -Z).
        #   RealSense standard: camera_link → optical = q(-0.5, 0.5, -0.5, 0.5)
        #     → optical Z lives along camera_link +X
        #   For optical Z → world -Z, camera_link +X must → world -Z
        #     → world_R_camera_link = Ry(+90°) * Rz(yaw)
        #
        #   Four candidate yaw values (0°/90°/180°/270°) all place bricks
        #   correctly on the table surface. The correct yaw depends on the
        #   physical camera rotation around its vertical axis.
        #
        #   DEFAULT: yaw=90° → q=(-0.5, 0.5, 0.5, 0.5)
        #     optical X (image right) → world +X
        #     optical Y (image down)  → world -Y
        #     optical Z (depth)       → world -Z  ✅
        #
        #   If bricks appear mirrored or rotated 90°/180° in X-Y, change
        #   camera_yaw_deg to 0, 180, or 270.
        # ---------------------------------------------------------------
        self.declare_parameter('camera_x',       1.0025)
        self.declare_parameter('camera_y',       0.0100)
        self.declare_parameter('camera_z',       1.92)

        # Yaw of camera around world Z (degrees). Tune if X/Y placement is
        # rotated relative to the real table. Try 0, 90, 180, 270.
        # Corresponding quaternions:
        #   0°  → qx= 0.0000, qy=0.7071, qz= 0.0000, qw=0.7071
        #  90°  → qx=-0.5000, qy=0.5000, qz= 0.5000, qw=0.5000
        # 180°  → qx=-0.7071, qy=0.0000, qz= 0.7071, qw=0.0000
        # 270°  → qx=-0.5000, qy=-0.500, qz= 0.5000, qw=-0.500
        self.declare_parameter('camera_yaw_deg', 90.0)

        # ---------------------------------------------------------------
        # Camera intrinsics (from /camera/camera/color/camera_info)
        # ---------------------------------------------------------------
        self.declare_parameter('fx', 609.70166015625)
        self.declare_parameter('fy', 609.23968505859)
        self.declare_parameter('cx', 336.19213867187)
        self.declare_parameter('cy', 250.18206787109)
        self.declare_parameter('depth_scale', 0.001)   # 16UC1 mm → m

        # ---------------------------------------------------------------
        # Z correction
        # world_z = cam_z - depth = 1.9035 - 0.712 = 1.1915
        # table surface = 1.100 m  → correction = -0.0915
        # ---------------------------------------------------------------
        self.declare_parameter('z_offset_correction', -0.091)

        # Depth patch size for robust depth sampling (NxN median)
        self.declare_parameter('depth_patch_size', 5)

        # ── read params ───────────────────────────────────────────────
        self.target_frame  = self.get_parameter('target_frame').value
        self.base_path     = self.get_parameter('mesh_base_path').value
        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value
        self.depth_scale   = self.get_parameter('depth_scale').value
        self.z_correction  = self.get_parameter('z_offset_correction').value
        self.patch         = self.get_parameter('depth_patch_size').value
        self.yaw_deg       = self.get_parameter('camera_yaw_deg').value

        # ============================================
        # TF SETUP
        # ============================================
        self.tf_buffer          = tf2_ros.Buffer()
        self.tf_listener        = tf2_ros.TransformListener(self.tf_buffer, self)
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self._publish_camera_tf()

        # ============================================
        # SUBSCRIBERS (time-synced YOLO + depth)
        # ============================================
        self.det_sub   = Subscriber(self, Detection2DArray, '/yolo/detections')
        self.depth_sub = Subscriber(self, Image,
                            '/camera/camera/aligned_depth_to_color/image_raw')
        self.sync = ApproximateTimeSynchronizer(
            [self.det_sub, self.depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.detections_callback)

        # ============================================
        # PUBLISHER
        # ============================================
        self.scene_pub = self.create_publisher(
            PlanningScene, '/planning_scene', 10)

        self.get_logger().info("✅ Real2Sim Bridge Started")
        self.get_logger().info(
            f"   camera pos   : ({self.get_parameter('camera_x').value:.4f}, "
            f"{self.get_parameter('camera_y').value:.4f}, "
            f"{self.get_parameter('camera_z').value:.4f})")
        self.get_logger().info(
            f"   camera yaw   : {self.yaw_deg}° "
            f"(change camera_yaw_deg if X/Y is wrong: try 0/90/180/270)")
        self.get_logger().info(
            f"   z_correction : {self.z_correction:.4f} m  "
            f"→ table Z = {1.9035 - 0.712 + self.z_correction:.3f} m")

    # ==========================================================
    # PUBLISH world → camera_link
    # ==========================================================
    def _publish_camera_tf(self):
        """
        Publishes the one missing TF link that connects the sim tree to
        the rosbag camera chain:

          sim  : world → abb_table → ...robots
          rosbag: camera_link → camera_color_frame → camera_color_optical_frame

        Rotation encodes a top-down camera with configurable yaw:
          Ry(+90°) * Rz(yaw)  →  optical Z maps to world -Z (downward) ✅
        """
        yaw = np.radians(self.yaw_deg)

        # Ry(+90°) * Rz(yaw)  — makes optical Z point world -Z
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        Ry90 = np.array([[0,0,1],[0,1,0],[-1,0,0]])
        roll = np.radians(90.0) 
        cr, sr = np.cos(roll), np.sin(roll)
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R = Ry90 @ Rz @ Rx

        # Convert to quaternion
        trace = R[0,0]+R[1,1]+R[2,2]
        if trace > 0:
            s2 = 0.5/np.sqrt(trace+1)
            qw = 0.25/s2
            qx = (R[2,1]-R[1,2])*s2
            qy = (R[0,2]-R[2,0])*s2
            qz = (R[1,0]-R[0,1])*s2
        elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
            s2 = 2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2])
            qw=(R[2,1]-R[1,2])/s2; qx=0.25*s2
            qy=(R[0,1]+R[1,0])/s2; qz=(R[0,2]+R[2,0])/s2
        elif R[1,1]>R[2,2]:
            s2 = 2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2])
            qw=(R[0,2]-R[2,0])/s2; qx=(R[0,1]+R[1,0])/s2
            qy=0.25*s2; qz=(R[1,2]+R[2,1])/s2
        else:
            s2 = 2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1])
            qw=(R[1,0]-R[0,1])/s2; qx=(R[0,2]+R[2,0])/s2
            qy=(R[1,2]+R[2,1])/s2; qz=0.25*s2

        t = TransformStamped()
        t.header.stamp    = self.get_clock().now().to_msg()
        t.header.frame_id = self.target_frame
        t.child_frame_id  = 'camera_color_optical_frame'
        t.transform.translation.x = self.get_parameter('camera_x').value
        t.transform.translation.y = self.get_parameter('camera_y').value
        t.transform.translation.z = self.get_parameter('camera_z').value
        t.transform.rotation.x = float(qx)
        t.transform.rotation.y = float(qy)
        t.transform.rotation.z = float(qz)
        t.transform.rotation.w = float(qw)
        self.static_broadcaster.sendTransform(t)

        self.get_logger().info(
            f"📡 world→camera_link: "
            f"pos=[{t.transform.translation.x:.4f}, "
            f"{t.transform.translation.y:.4f}, "
            f"{t.transform.translation.z:.4f}]  "
            f"q=[{qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f}]")

    # ==========================================================
    # DEPTH LOOKUP  (5×5 median patch, mm→m)
    # ==========================================================
    def _get_depth(self, depth_img: Image, u: int, v: int):
        h, w = depth_img.height, depth_img.width
        raw  = np.frombuffer(depth_img.data, dtype=np.uint16).reshape((h, w))
        half = self.patch // 2
        u0,u1 = max(0,u-half), min(w,u+half+1)
        v0,v1 = max(0,v-half), min(h,v+half+1)
        valid = raw[v0:v1, u0:u1].astype(np.float32)
        valid = valid[valid > 0]
        return float(np.median(valid)) * self.depth_scale if valid.size else None

    # ==========================================================
    # PIXEL → OPTICAL FRAME 3-D
    # ==========================================================
    def _deproject(self, u, v, depth):
        return ((u-self.cx)*depth/self.fx,
                (v-self.cy)*depth/self.fy,
                depth)

    # ==========================================================
    # MESH HELPERS
    # ==========================================================
    def _load_mesh(self, path):
        m = trimesh.load(path)
        m.apply_scale(0.001)
        msg = Mesh()
        for v in m.vertices:
            p = Point(); p.x,p.y,p.z = float(v[0]),float(v[1]),float(v[2])
            msg.vertices.append(p)
        for f in m.faces:
            t = MeshTriangle()
            t.vertex_indices = [int(f[0]),int(f[1]),int(f[2])]
            msg.triangles.append(t)
        return msg

    def _mesh_path(self, cls):
        t = CLASS_TO_TYPE.get(cls.upper())
        if t is None: return None
        d = {0:("I_brick","I_brick.STL"),
             1:("L_brick","L_brick.STL"),
             2:("T_brick","T_brick.STL")}
        return os.path.join(self.base_path, *d[t])

    # ==========================================================
    # MAIN CALLBACK
    # ==========================================================
    def detections_callback(self, det_msg: Detection2DArray,
                             depth_msg: Image):
        if not det_msg.detections:
            return

        planning_scene         = PlanningScene()
        planning_scene.is_diff = True
        objs = []

        for i, det in enumerate(det_msg.detections):
            if not det.results:
                continue

            best  = max(det.results, key=lambda r: r.hypothesis.score)
            cls   = best.hypothesis.class_id.strip().upper()
            score = best.hypothesis.score
            mesh_path = self._mesh_path(cls)
            if mesh_path is None:
                self.get_logger().warn(f"  Unknown class '{cls}'")
                continue

            # ── pixel centre ──────────────────────────────────
            u = int(round(det.bbox.center.position.x))
            v = int(round(det.bbox.center.position.y))

            # ── depth ────────────────────────────────────────
            depth = self._get_depth(depth_msg, u, v)
            if not depth or depth < 0.01:
                self.get_logger().warn(
                    f"  Brick {i} ({cls}): bad depth at ({u},{v})")
                continue

            # ── back-project → optical frame ──────────────────
            ox, oy, oz = self._deproject(u, v, depth)

            # ── yaw from bbox rotation ────────────────────────
            theta = det.bbox.center.theta
            ps = PoseStamped()
            ps.header.frame_id = depth_msg.header.frame_id             
            ps.header.stamp    = det_msg.header.stamp
            ps.pose.position.x = ox
            ps.pose.position.y = oy
            ps.pose.position.z = oz
            ps.pose.orientation.x = float(np.cos(theta/2))
            ps.pose.orientation.y = float(np.sin(theta/2))
            ps.pose.orientation.z = 0.0
            ps.pose.orientation.w = 0.0

            # ── transform → world ─────────────────────────────
            try:
                wp = self.tf_buffer.transform(
                    ps, self.target_frame,
                    timeout=rclpy.duration.Duration(seconds=1.0)).pose
            except Exception as e:
                self.get_logger().warn(f"  TF failed brick {i}: {e}")
                continue

            # ── Z correction ──────────────────────────────────
            # world_z = cam_z - depth (top-down camera, depth→world -Z)
            # Apply fixed offset to land on table surface (1.100 m)
            wp.position.z += self.z_correction

            # ── mesh ─────────────────────────────────────────
            try:
                mesh = self._load_mesh(mesh_path)
            except Exception as e:
                self.get_logger().error(f"  Mesh load failed: {e}")
                continue

            obj = CollisionObject()
            obj.id = f"brick_{i}"
            obj.header.frame_id = self.target_frame
            obj.meshes.append(mesh)
            obj.mesh_poses.append(wp)
            obj.operation = CollisionObject.ADD
            objs.append(obj)

            self.get_logger().info(
                f"  🧱 {cls} score={score:.2f} depth={depth:.3f}m → "
                f"world x={wp.position.x:.3f} y={wp.position.y:.3f} "
                f"z={wp.position.z:.3f}")

        planning_scene.world.collision_objects = objs
        self.scene_pub.publish(planning_scene)
        self.get_logger().info(
            f"✅ Published {len(objs)} objects to MoveIt")


def main(args=None):
    rclpy.init(args=args)
    node = Real2SimBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()