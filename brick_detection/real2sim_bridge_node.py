#!/usr/bin/env python3
"""
real2sim_bridge.py
==================
Bridges real-world YOLO brick detections into the simulation:
  • MoveIt / RViz  — publishes CollisionObject to /planning_scene
  • Gazebo (GZ Sim) — calls native GZ Transport service via `gz service` CLI

KEY BEHAVIOURS
--------------
1. Each brick is spawned ONCE, at the FIRST position it is detected.
   Subsequent detections of the same tracking-ID are silently ignored.
2. Spawning is dual: both MoveIt collision scene AND Gazebo world.

WHY gz CLI INSTEAD OF A ROS 2 SERVICE?
---------------------------------------
GZ Sim (Gazebo Harmonic) exposes /world/<n>/create as a *native GZ Transport*
service, NOT a ROS 2 service.  ros_gz_interfaces/srv/SpawnEntity does not
exist in Harmonic.  The correct way to reach it from Python is:

    gz service -s /world/<n>/create
               --reqtype  gz.msgs.EntityFactory
               --reptype  gz.msgs.Boolean
               --timeout  2000
               --req      '<proto payload>'

This is exactly what ros_gz_sim's `create` executable does internally.
We replicate it here with subprocess so no extra packages are needed.

DEPENDENCIES
------------
  gz  (GZ Sim CLI — present when Gazebo Harmonic is installed and sourced)
  trimesh             (pip install trimesh)
  message_filters     (standard ROS 2)

PARAMETERS (all have sensible defaults — override in a launch file)
----------
  target_frame          world
  mesh_base_path        ~/collab_ws/src/.../dual_arms/models
  camera_x/y/z          1.0025 / 0.01 / 1.92
  camera_yaw_deg        90.0
  fx / fy / cx / cy     RealSense defaults
  depth_scale           0.001
  z_offset_correction   -0.091
  depth_patch_size      5
  gazebo_world_name     empty   (used in /world/<n>/create)
"""

import os
import math
import shutil
import subprocess
import threading
import numpy as np
import trimesh

import rclpy
from rclpy.node import Node
import rclpy.duration

import tf2_ros
import tf2_geometry_msgs                       # noqa — registers transform support

from message_filters import Subscriber, ApproximateTimeSynchronizer

from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import Mesh, MeshTriangle


# ── YOLO class id -> internal type index ─────────────────────────────────────
CLASS_TO_TYPE = {"I": 0, "L": 1, "T": 2}

# ── Mesh sub-folder and filename per type ────────────────────────────────────
MESH_FILES = {
    0: ("I_brick", "I_brick.STL"),
    1: ("L_brick", "L_brick.STL"),
    2: ("T_brick", "T_brick.STL"),
}


# =============================================================================
class Real2SimBridge(Node):
    # -------------------------------------------------------------------------
    def __init__(self):
        super().__init__("real2sim_bridge")

        # ── parameters ───────────────────────────────────────────────────────
        self.declare_parameter("target_frame", "world")
        self.declare_parameter(
            "mesh_base_path",
            os.path.expanduser(
                "~/collab_ws/src/dual_arms_packages/dual_arms/models"
            ),
        )
        self.declare_parameter("camera_x",            1.0025)
        self.declare_parameter("camera_y",            0.0100)
        self.declare_parameter("camera_z",            1.92)
        self.declare_parameter("camera_yaw_deg",      90.0)
        self.declare_parameter("fx",                  609.70166015625)
        self.declare_parameter("fy",                  609.23968505859)
        self.declare_parameter("cx",                  336.19213867187)
        self.declare_parameter("cy",                  250.18206787109)
        self.declare_parameter("depth_scale",         0.001)
        self.declare_parameter("z_offset_correction", -0.091)
        self.declare_parameter("depth_patch_size",    5)
        self.declare_parameter("gazebo_world_name",   "empty")

        self.target_frame  = self.get_parameter("target_frame").value
        self.base_path     = self.get_parameter("mesh_base_path").value
        self.fx            = self.get_parameter("fx").value
        self.fy            = self.get_parameter("fy").value
        self.cx            = self.get_parameter("cx").value
        self.cy            = self.get_parameter("cy").value
        self.depth_scale   = self.get_parameter("depth_scale").value
        self.z_correction  = self.get_parameter("z_offset_correction").value
        self.patch         = self.get_parameter("depth_patch_size").value
        self.yaw_deg       = self.get_parameter("camera_yaw_deg").value
        self.gz_world      = self.get_parameter("gazebo_world_name").value

        # ── spawn-once registry ───────────────────────────────────────────────
        # Tracking-IDs added here are never processed again.
        self._spawned_ids: set = set()

        # ── TF ───────────────────────────────────────────────────────────────
        self.tf_buffer          = tf2_ros.Buffer()
        self.tf_listener        = tf2_ros.TransformListener(self.tf_buffer, self)
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self._publish_camera_tf()

        # ── time-synced subscribers ───────────────────────────────────────────
        self.det_sub   = Subscriber(self, Detection2DArray, "/yolo/detections")
        self.depth_sub = Subscriber(
            self, Image, "/camera/camera/aligned_depth_to_color/image_raw"
        )
        self.sync = ApproximateTimeSynchronizer(
            [self.det_sub, self.depth_sub], queue_size=10, slop=0.05
        )
        self.sync.registerCallback(self._detections_callback)

        # ── MoveIt / RViz publisher ───────────────────────────────────────────
        self.scene_pub = self.create_publisher(PlanningScene, "/planning_scene", 10)

        # ── Gazebo CLI check ──────────────────────────────────────────────────
        # Sets self._gz_available and self._gz_bin
        self._gz_bin       = "gz"
        self._gz_available = self._check_gz_cli()

        self.get_logger().info("Real2SimBridge started (spawn-once mode)")
        self.get_logger().info(
            "   camera pos : ({:.4f}, {:.4f}, {:.4f})".format(
                self.get_parameter("camera_x").value,
                self.get_parameter("camera_y").value,
                self.get_parameter("camera_z").value,
            )
        )
        self.get_logger().info(
            "   camera yaw : {}  (try 0/90/180/270 if placement looks rotated)".format(
                self.yaw_deg
            )
        )
        self.get_logger().info(
            "   z_correction: {:.4f} m  | Gazebo world: '{}'".format(
                self.z_correction, self.gz_world
            )
        )

    # =========================================================================
    # GAZEBO CLI CHECK
    # =========================================================================
    def _check_gz_cli(self) -> bool:
        """
        Check whether the `gz` (or `ign`) binary is available.
        GZ Sim Harmonic does NOT expose /world/*/create as a ROS 2 service —
        it is a native GZ Transport service reachable only via `gz service`.
        """
        if shutil.which("gz") is not None:
            self._gz_bin = "gz"
            self.get_logger().info(
                "   Gazebo spawn : gz CLI found "
                "-> /world/{}/create".format(self.gz_world)
            )
            return True

        # Ignition fallback (Fortress / Garden)
        if shutil.which("ign") is not None:
            self._gz_bin = "ign"
            self.get_logger().info(
                "   Gazebo spawn : ign CLI found (Ignition fallback)"
            )
            return True

        self.get_logger().warn(
            "gz / ign CLI not found -- Gazebo spawning DISABLED.\n"
            "Source GZ Sim:  source /opt/ros/$ROS_DISTRO/setup.bash"
        )
        return False

    # =========================================================================
    # CAMERA TF
    # =========================================================================
    def _publish_camera_tf(self):
        """
        Static world -> camera_color_optical_frame.
        Rotation = Ry(+90) * Rz(yaw) * Rx(+90) so optical-Z maps to world -Z.
        """
        yaw    = np.radians(self.yaw_deg)
        c, s   = np.cos(yaw), np.sin(yaw)
        Rz     = np.array([[c, -s, 0], [s,  c, 0], [0, 0, 1]])
        Ry90   = np.array([[0,  0, 1], [0,  1, 0], [-1, 0, 0]])
        roll   = np.radians(90.0)
        cr, sr = np.cos(roll), np.sin(roll)
        Rx     = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R      = Ry90 @ Rz @ Rx
        qx, qy, qz, qw = _mat_to_quat(R)

        t = TransformStamped()
        t.header.stamp            = self.get_clock().now().to_msg()
        t.header.frame_id         = self.target_frame
        t.child_frame_id          = "camera_color_optical_frame"
        t.transform.translation.x = self.get_parameter("camera_x").value
        t.transform.translation.y = self.get_parameter("camera_y").value
        t.transform.translation.z = self.get_parameter("camera_z").value
        t.transform.rotation.x    = float(qx)
        t.transform.rotation.y    = float(qy)
        t.transform.rotation.z    = float(qz)
        t.transform.rotation.w    = float(qw)
        self.static_broadcaster.sendTransform(t)

        self.get_logger().info(
            "camera_color_optical_frame TF published  q=[{:.4f}, {:.4f}, {:.4f}, {:.4f}]".format(
                qx, qy, qz, qw
            )
        )

    # =========================================================================
    # DEPTH HELPERS
    # =========================================================================
    def _get_depth(self, depth_img: Image, u: int, v: int):
        h    = depth_img.height
        w    = depth_img.width
        raw  = np.frombuffer(depth_img.data, dtype=np.uint16).reshape((h, w))
        half = self.patch // 2
        u0, u1 = max(0, u - half), min(w, u + half + 1)
        v0, v1 = max(0, v - half), min(h, v + half + 1)
        patch  = raw[v0:v1, u0:u1].astype(np.float32)
        valid  = patch[patch > 0]
        return float(np.median(valid)) * self.depth_scale if valid.size else None

    def _deproject(self, u, v, depth):
        return (
            (u - self.cx) * depth / self.fx,
            (v - self.cy) * depth / self.fy,
            depth,
        )

    # =========================================================================
    # MESH HELPERS
    # =========================================================================
    def _load_mesh_msg(self, path: str) -> Mesh:
        m = trimesh.load(path)
        m.apply_scale(0.001)        # STL in mm -> metres
        msg = Mesh()
        for v in m.vertices:
            p = Point()
            p.x, p.y, p.z = float(v[0]), float(v[1]), float(v[2])
            msg.vertices.append(p)
        for f in m.faces:
            tri = MeshTriangle()
            tri.vertex_indices = [int(f[0]), int(f[1]), int(f[2])]
            msg.triangles.append(tri)
        return msg

    def _mesh_path(self, cls: str):
        idx = CLASS_TO_TYPE.get(cls.upper())
        if idx is None:
            return None
        folder, filename = MESH_FILES[idx]
        return os.path.join(self.base_path, folder, filename)

    # =========================================================================
    # RVIZ / MOVEIT SPAWN
    # =========================================================================
    def _spawn_in_rviz(self, brick_id: str, cls: str, world_pose, stamp):
        mesh_path = self._mesh_path(cls)
        if mesh_path is None or not os.path.exists(mesh_path):
            self.get_logger().warn("Mesh not found for RViz: {}".format(mesh_path))
            return
        try:
            mesh_msg = self._load_mesh_msg(mesh_path)
        except Exception as exc:
            self.get_logger().error("Mesh load error {}: {}".format(brick_id, exc))
            return

        col_obj                 = CollisionObject()
        col_obj.id              = brick_id
        col_obj.header.frame_id = self.target_frame
        col_obj.header.stamp    = stamp
        col_obj.meshes.append(mesh_msg)
        col_obj.mesh_poses.append(world_pose)
        col_obj.operation       = CollisionObject.ADD

        scene                         = PlanningScene()
        scene.is_diff                 = True
        scene.world.collision_objects = [col_obj]
        self.scene_pub.publish(scene)

    # =========================================================================
    # GAZEBO SPAWN via gz service CLI
    # =========================================================================
    def _spawn_in_gazebo(self, brick_id: str, cls: str, world_pose):
        """
        Call /world/<name>/create via `gz service` in a daemon thread.

        The EntityFactory protobuf request is passed as a text-format string.
        The SDF is embedded in the `sdf` field of the proto.

        Key SDF requirements for GZ Sim:
          - <pose> must be a child of <model>, not <link>
          - pose format is: "x y z roll pitch yaw"
          - mesh URI must be  file:///absolute/path/to/mesh.STL
          - scale 0.001 converts mm STL -> metres
        """
        if not self._gz_available:
            return

        mesh_path = self._mesh_path(cls)
        if mesh_path is None or not os.path.exists(mesh_path):
            self.get_logger().warn(
                "Mesh not found for Gazebo: {}".format(mesh_path)
            )
            return

        px = world_pose.position.x
        py = world_pose.position.y
        pz = world_pose.position.z
        roll, pitch, yaw = _quat_to_rpy(
            world_pose.orientation.x,
            world_pose.orientation.y,
            world_pose.orientation.z,
            world_pose.orientation.w,
        )

        # Build inline SDF ─────────────────────────────────────────────────
        sdf = (
            "<?xml version='1.0'?>"
            "<sdf version='1.9'>"
            "<model name='{name}'>"
            "<static>true</static>"
            "<pose>{px} {py} {pz} {roll} {pitch} {yaw}</pose>"
            "<link name='link'>"
            "<visual name='visual'>"
            "<geometry><mesh>"
            "<uri>file://{mesh}</uri>"
            "<scale>0.001 0.001 0.001</scale>"
            "</mesh></geometry>"
            "<material>"
            "<ambient>0.2 0.6 1.0 1</ambient>"
            "<diffuse>0.2 0.6 1.0 1</diffuse>"
            "</material>"
            "</visual>"
            "<collision name='collision'>"
            "<geometry><mesh>"
            "<uri>file://{mesh}</uri>"
            "<scale>0.001 0.001 0.001</scale>"
            "</mesh></geometry>"
            "</collision>"
            "</link>"
            "</model>"
            "</sdf>"
        ).format(
            name=brick_id,
            px=px, py=py, pz=pz,
            roll=roll, pitch=pitch, yaw=yaw,
            mesh=mesh_path,
        )

        # EntityFactory proto text-format payload ──────────────────────────
        # Single quotes inside the SDF would break the proto string, so we
        # replace them with double quotes (SDF allows both).
        sdf_for_proto = sdf.replace("'", '"')
        proto_req = "sdf: '{}' name: '{}'".format(sdf_for_proto, brick_id)

        cmd = [
            self._gz_bin, "service",
            "-s", "/world/{}/create".format(self.gz_world),
            "--reqtype",  "gz.msgs.EntityFactory",
            "--reptype",  "gz.msgs.Boolean",
            "--timeout",  "2000",
            "--req",      proto_req,
        ]

        node_ref = self  # capture for thread closure

        def _run():
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if result.returncode == 0:
                    node_ref.get_logger().info(
                        "   Gazebo: spawned '{}'".format(brick_id)
                    )
                else:
                    node_ref.get_logger().warn(
                        "   Gazebo spawn FAILED for '{}':\n"
                        "     stdout: {}\n"
                        "     stderr: {}".format(
                            brick_id,
                            result.stdout.strip(),
                            result.stderr.strip(),
                        )
                    )
            except subprocess.TimeoutExpired:
                node_ref.get_logger().warn(
                    "   Gazebo spawn timed out for '{}'".format(brick_id)
                )
            except Exception as exc:
                node_ref.get_logger().error(
                    "   Gazebo spawn exception for '{}': {}".format(brick_id, exc)
                )

        threading.Thread(target=_run, daemon=True).start()

    # =========================================================================
    # MAIN CALLBACK
    # =========================================================================
    def _detections_callback(
        self, det_msg: Detection2DArray, depth_msg: Image
    ):
        if not det_msg.detections:
            return

        for det in det_msg.detections:
            if not det.results:
                continue

            # ── tracking ID ───────────────────────────────────────────────
            tracking_id = str(det.id).strip()

            # ── spawn-once gate ───────────────────────────────────────────
            if tracking_id in self._spawned_ids:
                continue

            # ── class ─────────────────────────────────────────────────────
            best = max(det.results, key=lambda r: r.hypothesis.score)
            cls  = best.hypothesis.class_id.strip().upper()
            if CLASS_TO_TYPE.get(cls) is None:
                self.get_logger().warn("Unknown class '{}' -- skipping".format(cls))
                continue

            # ── pixel centre ──────────────────────────────────────────────
            u = int(round(det.bbox.center.position.x))
            v = int(round(det.bbox.center.position.y))

            # ── depth ─────────────────────────────────────────────────────
            depth = self._get_depth(depth_msg, u, v)
            if not depth or depth < 0.01:
                self.get_logger().warn(
                    "Brick '{}' ({}): bad depth at ({},{}) -- retrying".format(
                        tracking_id, cls, u, v
                    )
                )
                continue   # retry next frame; do NOT add to _spawned_ids

            # ── back-project -> optical frame ─────────────────────────────
            ox, oy, oz = self._deproject(u, v, depth)
            theta      = det.bbox.center.theta

            ps                    = PoseStamped()
            ps.header.frame_id    = depth_msg.header.frame_id
            ps.header.stamp       = det_msg.header.stamp
            ps.pose.position.x    = ox
            ps.pose.position.y    = oy
            ps.pose.position.z    = oz
            ps.pose.orientation.x = float(np.cos(theta / 2))
            ps.pose.orientation.y = float(np.sin(theta / 2))
            ps.pose.orientation.z = 0.0
            ps.pose.orientation.w = 0.0

            # ── transform -> world ────────────────────────────────────────
            try:
                world_ps = self.tf_buffer.transform(
                    ps,
                    self.target_frame,
                    timeout=rclpy.duration.Duration(seconds=1.0),
                )
            except Exception as exc:
                self.get_logger().warn(
                    "TF failed for '{}': {} -- retrying".format(tracking_id, exc)
                )
                continue

            world_ps.pose.position.z += self.z_correction
            world_pose = world_ps.pose

            brick_id = "brick_{}_{}".format(cls.lower(), tracking_id)

            self.get_logger().info(
                "First detection -> spawning '{}'  "
                "x={:.3f} y={:.3f} z={:.3f}".format(
                    brick_id,
                    world_pose.position.x,
                    world_pose.position.y,
                    world_pose.position.z,
                )
            )

            # ── spawn in RViz / MoveIt ────────────────────────────────────
            self._spawn_in_rviz(brick_id, cls, world_pose, det_msg.header.stamp)

            # ── spawn in Gazebo ───────────────────────────────────────────
            self._spawn_in_gazebo(brick_id, cls, world_pose)

            # ── lock this ID forever ──────────────────────────────────────
            self._spawned_ids.add(tracking_id)


# =============================================================================
# PURE MATHS HELPERS
# =============================================================================

def _mat_to_quat(R: np.ndarray):
    """3x3 rotation matrix -> (qx, qy, qz, qw)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s  = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s  = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s  = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s  = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return qx, qy, qz, qw


def _quat_to_rpy(qx, qy, qz, qw):
    """Quaternion -> (roll, pitch, yaw) in radians."""
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp  = 2.0 * (qw * qy - qz * qx)
    pitch = (
        math.copysign(math.pi / 2.0, sinp)
        if abs(sinp) >= 1.0
        else math.asin(sinp)
    )

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# =============================================================================
# ENTRY POINT
# =============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = Real2SimBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()