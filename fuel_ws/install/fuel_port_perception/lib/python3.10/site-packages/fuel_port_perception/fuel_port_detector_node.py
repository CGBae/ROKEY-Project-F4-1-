#!/usr/bin/env python3
"""
fuel_port_detector_node.py

다른 PC 또는 같은 PC에서 실행하는 ROS2 perception node입니다.
입력:
  - /rgb          sensor_msgs/Image
  - /depth        sensor_msgs/Image
  - /camera_info  sensor_msgs/CameraInfo
선택 입력:
  - TF: camera frame -> world/base frame 변환이 있으면 world/base 좌표도 발행

출력:
  - /fuel_port_point_camera  geometry_msgs/PointStamped, frame_id=camera optical frame
  - /fuel_port_point         geometry_msgs/PointStamped, frame_id=target_frame. TF 가능할 때만 publish
  - /fuel_port_detected      std_msgs/Bool

기능:
  - 빨간색 임시 주유구 모델을 HSV threshold로 검출
  - 중심 pixel, depth, camera 좌표, target/world 좌표를 계산
  - 화면 상단에 수치 overlay 표시
  - 터미널에도 수치를 주기적으로 출력
  - 최근 N프레임의 좌표가 안정적일 때만 /fuel_port_point를 publish
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge

import message_filters
import tf2_ros
try:
    import tf2_geometry_msgs  # noqa: F401  # do_transform_point 등록용
except Exception:
    tf2_geometry_msgs = None


@dataclass
class DetectionResult:
    detected: bool
    u: int = -1
    v: int = -1
    area: float = 0.0
    depth_m: float = float("nan")
    point_camera: Optional[np.ndarray] = None
    point_target: Optional[np.ndarray] = None
    stable: bool = False
    stable_std_m: float = float("nan")
    contour: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    reason: str = ""


class FuelPortDetectorNode(Node):
    def __init__(self):
        super().__init__("fuel_port_detector_node")
        self.bridge = CvBridge()

        # -----------------------------
        # ROS parameters
        # -----------------------------
        self.declare_parameter("rgb_topic", "/rgb")
        self.declare_parameter("depth_topic", "/depth")
        self.declare_parameter("camera_info_topic", "/camera_info")
        self.declare_parameter("target_frame", "world")
        self.declare_parameter("publish_only_when_stable", True)
        self.declare_parameter("required_stable_frames", 5)
        self.declare_parameter("stable_std_threshold_m", 0.01)
        self.declare_parameter("min_contour_area", 80.0)
        self.declare_parameter("depth_patch_radius", 4)
        self.declare_parameter("show_window", True)
        self.declare_parameter("print_every_n_frames", 10)
        self.declare_parameter("use_tf", True)

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.publish_only_when_stable = bool(self.get_parameter("publish_only_when_stable").value)
        self.required_stable_frames = int(self.get_parameter("required_stable_frames").value)
        self.stable_std_threshold_m = float(self.get_parameter("stable_std_threshold_m").value)
        self.min_contour_area = float(self.get_parameter("min_contour_area").value)
        self.depth_patch_radius = int(self.get_parameter("depth_patch_radius").value)
        self.show_window = bool(self.get_parameter("show_window").value)
        self.print_every_n_frames = int(self.get_parameter("print_every_n_frames").value)
        self.use_tf = bool(self.get_parameter("use_tf").value)

        # -----------------------------
        # QoS: image topic에는 BEST_EFFORT가 자주 쓰임
        # -----------------------------
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        pub_qos = QoSProfile(depth=10)

        # message_filters로 RGB/Depth/CameraInfo 시간 동기화
        self.rgb_sub = message_filters.Subscriber(self, Image, self.rgb_topic, qos_profile=image_qos)
        self.depth_sub = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=image_qos)
        self.info_sub = message_filters.Subscriber(self, CameraInfo, self.camera_info_topic, qos_profile=image_qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub],
            queue_size=10,
            slop=0.08,
        )
        self.sync.registerCallback(self.synced_callback)

        self.pub_camera_point = self.create_publisher(PointStamped, "/fuel_port_point_camera", pub_qos)
        self.pub_target_point = self.create_publisher(PointStamped, "/fuel_port_point", pub_qos)
        self.pub_detected = self.create_publisher(Bool, "/fuel_port_detected", pub_qos)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.camera_buffer = deque(maxlen=self.required_stable_frames)
        self.target_buffer = deque(maxlen=self.required_stable_frames)
        self.frame_count = 0

        self.get_logger().info("Fuel port detector node started")
        self.get_logger().info(f"  rgb_topic         = {self.rgb_topic}")
        self.get_logger().info(f"  depth_topic       = {self.depth_topic}")
        self.get_logger().info(f"  camera_info_topic = {self.camera_info_topic}")
        self.get_logger().info(f"  target_frame      = {self.target_frame}")
        self.get_logger().info("  detection target  = red temporary fuel-port model")

    # ========================================================
    # Image callback
    # ========================================================
    def synced_callback(self, rgb_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        self.frame_count += 1

        try:
            bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge conversion failed: {exc}")
            return

        result = self.process_frame(bgr, depth_raw, info_msg, rgb_msg.header.frame_id, rgb_msg.header.stamp)

        self.pub_detected.publish(Bool(data=bool(result.detected)))

        # 안정적일 때만 또는 사용자가 설정한 경우 즉시 publish
        should_publish = result.detected and result.point_camera is not None
        if self.publish_only_when_stable:
            should_publish = should_publish and result.stable

        if should_publish:
            p_cam_msg = self.make_point_msg(result.point_camera, rgb_msg.header.frame_id, rgb_msg.header.stamp)
            self.pub_camera_point.publish(p_cam_msg)

            if result.point_target is not None:
                p_target_msg = self.make_point_msg(result.point_target, self.target_frame, rgb_msg.header.stamp)
                self.pub_target_point.publish(p_target_msg)

        self.print_numeric(result, rgb_msg.header.frame_id)

        if self.show_window:
            vis = self.draw_overlay(bgr, result, rgb_msg.header.frame_id)
            cv2.imshow("fuel_port_detector: RGB view", vis)
            if result.mask is not None:
                cv2.imshow("fuel_port_detector: red mask", result.mask)
            cv2.waitKey(1)

    # ========================================================
    # Core processing
    # ========================================================
    def process_frame(self,
                      bgr: np.ndarray,
                      depth_raw: np.ndarray,
                      info_msg: CameraInfo,
                      camera_frame_id: str,
                      stamp) -> DetectionResult:
        center, contour, mask, area = self.detect_red_port(bgr)
        if center is None:
            self.camera_buffer.clear()
            self.target_buffer.clear()
            return DetectionResult(False, mask=mask, reason="red target not detected")

        u, v = center
        depth_m = self.get_depth_median(depth_raw, u, v, self.depth_patch_radius)
        if depth_m is None or not np.isfinite(depth_m) or depth_m <= 0.0:
            self.camera_buffer.clear()
            self.target_buffer.clear()
            return DetectionResult(False, u=u, v=v, area=area, mask=mask, contour=contour, reason="invalid depth")

        point_camera = self.pixel_to_camera_point(u, v, depth_m, info_msg)
        point_target = None

        if self.use_tf:
            point_target = self.try_transform_point(point_camera, camera_frame_id, stamp)

        # 안정화 판정은 target 좌표가 있으면 target 좌표 기준, 없으면 camera 좌표 기준
        stable_point = point_target if point_target is not None else point_camera
        stable, std_norm = self.update_stability(stable_point)

        return DetectionResult(
            detected=True,
            u=u,
            v=v,
            area=area,
            depth_m=depth_m,
            point_camera=point_camera,
            point_target=point_target,
            stable=stable,
            stable_std_m=std_norm,
            contour=contour,
            mask=mask,
            reason="ok",
        )

    def detect_red_port(self, bgr: np.ndarray):
        """빨간색 영역을 HSV threshold로 검출하고 가장 큰 contour 중심을 반환."""
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # 빨간색은 hue가 0 근처와 180 근처로 나뉨
        lower_red1 = np.array([0, 80, 80], dtype=np.uint8)
        upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
        lower_red2 = np.array([170, 80, 80], dtype=np.uint8)
        upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None, None, mask, 0.0

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area < self.min_contour_area:
            return None, contour, mask, area

        M = cv2.moments(contour)
        if abs(M["m00"]) < 1e-9:
            return None, contour, mask, area

        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])
        return (u, v), contour, mask, area

    def get_depth_median(self, depth_raw: np.ndarray, u: int, v: int, radius: int) -> Optional[float]:
        h, w = depth_raw.shape[:2]
        u0 = max(0, u - radius)
        u1 = min(w, u + radius + 1)
        v0 = max(0, v - radius)
        v1 = min(h, v + radius + 1)
        patch = depth_raw[v0:v1, u0:u1]

        if patch.size == 0:
            return None

        patch = np.asarray(patch).astype(np.float32)

        # 16UC1이면 보통 mm, 32FC1이면 보통 m
        # 값 범위로 한 번 더 방어: 10보다 크면 mm일 가능성이 높다.
        finite = patch[np.isfinite(patch)]
        if finite.size == 0:
            return None
        median_val = float(np.median(finite))
        if median_val > 10.0:
            patch = patch * 0.001

        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def pixel_to_camera_point(self, u: int, v: int, depth_m: float, info_msg: CameraInfo) -> np.ndarray:
        fx = float(info_msg.k[0])
        fy = float(info_msg.k[4])
        cx = float(info_msg.k[2])
        cy = float(info_msg.k[5])

        if abs(fx) < 1e-9 or abs(fy) < 1e-9:
            raise RuntimeError("CameraInfo intrinsic K is invalid")

        z = float(depth_m)
        x = (float(u) - cx) * z / fx
        y = (float(v) - cy) * z / fy
        return np.array([x, y, z], dtype=float)

    def try_transform_point(self, point_camera: np.ndarray, camera_frame_id: str, stamp) -> Optional[np.ndarray]:
        if not self.use_tf:
            return None
        try:
            msg = self.make_point_msg(point_camera, camera_frame_id, stamp)
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                camera_frame_id,
                rclpy.time.Time.from_msg(stamp),
                timeout=rclpy.duration.Duration(seconds=0.03),
            )
            transformed = self.tf_buffer.transform(msg, self.target_frame, timeout=rclpy.duration.Duration(seconds=0.03))
            return np.array([
                transformed.point.x,
                transformed.point.y,
                transformed.point.z,
            ], dtype=float)
        except Exception:
            # TF가 없으면 camera 좌표만 사용한다. 너무 자주 로그를 찍지 않기 위해 print_numeric에서 상태 표시.
            return None

    def update_stability(self, point: np.ndarray) -> Tuple[bool, float]:
        self.camera_buffer.append(np.asarray(point, dtype=float))
        if len(self.camera_buffer) < self.required_stable_frames:
            return False, float("nan")
        arr = np.array(self.camera_buffer)
        std_xyz = np.std(arr, axis=0)
        std_norm = float(np.linalg.norm(std_xyz))
        return std_norm < self.stable_std_threshold_m, std_norm

    @staticmethod
    def make_point_msg(point: np.ndarray, frame_id: str, stamp) -> PointStamped:
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.point.x = float(point[0])
        msg.point.y = float(point[1])
        msg.point.z = float(point[2])
        return msg

    # ========================================================
    # Visualization / terminal print
    # ========================================================
    def print_numeric(self, result: DetectionResult, camera_frame_id: str):
        if self.frame_count % max(1, self.print_every_n_frames) != 0:
            return

        if not result.detected:
            self.get_logger().info(f"[frame {self.frame_count}] detected=False reason={result.reason}")
            return

        cam_txt = "None"
        if result.point_camera is not None:
            p = result.point_camera
            cam_txt = f"({p[0]: .4f}, {p[1]: .4f}, {p[2]: .4f})"

        tgt_txt = "None"
        if result.point_target is not None:
            p = result.point_target
            tgt_txt = f"({p[0]: .4f}, {p[1]: .4f}, {p[2]: .4f})"

        self.get_logger().info(
            f"[frame {self.frame_count}] "
            f"uv=({result.u:4d},{result.v:4d}) "
            f"depth={result.depth_m:.4f}m "
            f"area={result.area:.1f} "
            f"camera[{camera_frame_id}]={cam_txt} "
            f"{self.target_frame}={tgt_txt} "
            f"stable={result.stable} std={result.stable_std_m:.5f}"
        )

    def draw_overlay(self, bgr: np.ndarray, result: DetectionResult, camera_frame_id: str) -> np.ndarray:
        vis = bgr.copy()

        # 상단 검정 반투명 박스
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (vis.shape[1], 150), (0, 0, 0), -1)
        vis = cv2.addWeighted(overlay, 0.55, vis, 0.45, 0)

        if result.contour is not None:
            cv2.drawContours(vis, [result.contour], -1, (0, 255, 255), 2)

        if result.detected:
            cv2.circle(vis, (result.u, result.v), 6, (0, 255, 0), -1)
            cv2.drawMarker(vis, (result.u, result.v), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)

        lines = []
        lines.append(f"detected={result.detected}  stable={result.stable}  reason={result.reason}")
        if result.detected:
            lines.append(f"pixel u,v=({result.u},{result.v})  depth={result.depth_m:.4f} m  area={result.area:.1f}")
            if result.point_camera is not None:
                p = result.point_camera
                lines.append(f"camera[{camera_frame_id}] xyz=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m")
            else:
                lines.append(f"camera[{camera_frame_id}] xyz=None")
            if result.point_target is not None:
                p = result.point_target
                lines.append(f"{self.target_frame} xyz=({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m")
            else:
                lines.append(f"{self.target_frame} xyz=None  TF unavailable or disabled")
            lines.append(f"stability std={result.stable_std_m:.5f} m / threshold={self.stable_std_threshold_m:.5f}")
        else:
            lines.append("waiting for red fuel-port target...")

        y = 24
        for line in lines:
            cv2.putText(vis, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
            y += 28
        return vis


def main():
    rclpy.init()
    node = FuelPortDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_window:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
