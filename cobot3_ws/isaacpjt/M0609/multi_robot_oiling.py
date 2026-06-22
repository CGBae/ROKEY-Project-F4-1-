from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String

import numpy as np
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator


_THIS_DIR = Path(__file__).resolve().parent
RMPFLOW_DIR = str(_THIS_DIR / "rmpflow")
if RMPFLOW_DIR not in sys.path:
    sys.path.insert(0, RMPFLOW_DIR)

from m0609_rmpflow_controller import RMPFlowController

# ╔══════════════════════════════════════════════════════════════╗
# ║  A. 두 로봇이 공유하는 환경 파라미터                            ║
# ╚══════════════════════════════════════════════════════════════╝
USD_PATH        = str(_THIS_DIR / "Collected_nozzletip_project/nozzletip_project.usd")
# m0609_A, m0609_B, fuel_door(Revolution Joint 포함), fuel_cap, fuel_port_hole이
# 이미 이 USD 안에 모델링되어 있다고 가정한다 (Isaac Sim 에디터에서 직접 구성).
EE_LINK_NAME    = "link_6"
GRIPPER_JOINTS  = ["finger_joint", "right_inner_knuckle_joint"]

DRIVE_STIFFNESS = 1e8
DRIVE_DAMPING   = 1e4
DRIVE_MAX_FORCE = 1e8

GRIPPER_OPEN    = [0.0, 0.0]
GRIPPER_CLOSE   = [0.5, 0.5]
GRIPPER_DELTA   = [-0.5, -0.5]

M0609_URDF_PATH           = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf")
M0609_DESCRIPTION_PATH    = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")

INITIAL_ARM_JOINT_DEG = {
    "joint_1": 10.0,
    "joint_2": -66.0,
    "joint_3": 150.0,
    "joint_4": 3.5,
    "joint_5": -75.0,
    "joint_6": -5.0,
}
INITIAL_GRIPPER_JOINTS = {
    "finger_joint": 0.0,
    "right_inner_knuckle_joint": 0.0,
}

# ╔══════════════════════════════════════════════════════════════╗
# ║  B. 로봇 A/B 배치                                               ║
# ╚══════════════════════════════════════════════════════════════╝
ROBOT_A_PRIM_PATH    = "/World/m0609_A"
ROBOT_A_BASE_WORLD   = np.array([0.0, 0.0, 1.0], dtype=float)
ROBOT_A_BASE_EULER_DEG = np.array([90.0, 0.0, 0.0], dtype=float)
ROBOT_A_BASE_ORIENTATION = euler_angles_to_quat(np.deg2rad(ROBOT_A_BASE_EULER_DEG))

ROBOT_B_PRIM_PATH    = "/World/m0609_B"
ROBOT_B_BASE_WORLD   = np.array([0.91, 0.0, 1.0], dtype=float)
ROBOT_B_BASE_EULER_DEG = np.array([90.0, 0.0, 0.0], dtype=float)
ROBOT_B_BASE_ORIENTATION = euler_angles_to_quat(np.deg2rad(ROBOT_B_BASE_EULER_DEG))

# ╔══════════════════════════════════════════════════════════════╗
# ║  C. 씬 오브젝트 위치 / 이름                                     ║
# ╚══════════════════════════════════════════════════════════════╝
FUEL_DOOR_CENTER      = np.array([0.67315, -1.33891, 1.02293], dtype=float)
FUEL_CAP_CENTER       = np.array([0.67281, -1.42088, 1.06441], dtype=float)
FUEL_PORT_HOLE_CENTER = np.array([0.68002, -1.64065, 1.00525], dtype=float)

FUEL_DOOR_PRIM_NAME = "fuel_door"
FUEL_CAP_PRIM_NAME = "fuel_cap"
FUEL_PORT_HOLE_PRIM_NAME = "fuel_port_hole"
SCENE_SEARCH_ROOT = "/World"

# 벽면 기준 바깥 방향 / 삽입 방향. door/cap/hole이 같은 차체 벽면에 있다고 가정하고 공유한다.
# 실제 USD 배치가 다르면 이 두 값만 조정하면 된다.
_OUTWARD_ANGLE_DEG = 105.0
PORT_OUTWARD_NORMAL = np.array(
    [0.0, np.sin(np.deg2rad(_OUTWARD_ANGLE_DEG)), -np.cos(np.deg2rad(_OUTWARD_ANGLE_DEG))], dtype=float
)
INSERTION_DIRECTION = -PORT_OUTWARD_NORMAL

FUEL_PORT_DIAMETER = 0.10
FUEL_PORT_DEPTH = 0.10
INSERT_DISTANCE = FUEL_PORT_DEPTH / 2.0

VIRTUAL_NOZZLE_LENGTH = 0.65
VIRTUAL_NOZZLE_Z_OFFSET = -0.25

# ╔══════════════════════════════════════════════════════════════╗
# ║  D. 제어 파라미터 (prompt 지정값)                                ║
# ╚══════════════════════════════════════════════════════════════╝
PHYSICS_DT = 1.0 / 60.0
POSITION_TOLERANCE = 0.060
MAX_STEPS_PER_STAGE = 1200
PRINT_EVERY_N_STEPS = 20

DEFAULT_TARGET_SPEED = 0.060
NEAR_TARGET_SPEED    = 0.040
INSERT_TARGET_SPEED  = 0.020
RETREAT_TARGET_SPEED = 0.050

HOME_JOINT_SPEED_ALPHA = 0.012
HOME_JOINT_TOLERANCE   = 0.035
HOME_HOLD_STEPS        = 80

FAR_DISTANCE  = 0.28
MID_DISTANCE  = 0.18
NEAR_DISTANCE = 0.09

COVER_START_DEG = 30.0
COVER_MID_DEG   = 80.0
COVER_OPEN_DEG  = 130.0
DOOR_ANGLE_TOLERANCE_DEG = 8.0
# 힌지 축 방향 부호가 USD/PhysX 쪽과 반대로 측정될 수 있다.
# 리셋 직후 로그의 "door angle"이 COVER_START_DEG와 다르게 튀면 -1.0으로 바꿔서 맞춘다.
DOOR_ANGLE_SIGN = 1.0

CAP_JOINT6_UNSCREW_DEG = -360.0
CAP_JOINT6_SCREW_DEG   = 360.0
CAP_JOINT6_DEG_PER_STEP = 4.0

GRIPPER_ACTION_HOLD_STEPS = 30

USE_TARGET_ORIENTATION = True

CAMERA_PRIM_CANDIDATES = [
    "/World/wall/rsd455/RSD455",
    "/World/wall/rsd455/Camera",
    "/World/wall/rsd455/camera",
]
CAMERA_POINT_CONVENTION = "ros_optical"

REQUIRE_TARGET_LOCK = True
CONTROLLER_REQUIRED_LOCK_SAMPLES = 5
CONTROLLER_WORLD_STD_TOLERANCE = 0.025
SEARCH_GATE_HALF_EXTENT = np.array([0.35, 0.35, 0.18], dtype=float)
WAIT_LOCK_TIMEOUT_STEPS = 900

# ╔══════════════════════════════════════════════════════════════╗
# ║  E. ROS2 토픽 이름                                              ║
# ╚══════════════════════════════════════════════════════════════╝
TOPIC_COLOR_POSE = "/color_detector/pose"
TOPIC_COLOR_LOCK = "/color_detector/target_locked"
TOPIC_MODE_SWITCH = "/color_detector/mode_switch"
TOPIC_ROBOT_A_DONE = "/robot_a/done"
TOPIC_ROBOT_B_DONE = "/robot_b/done"


# ============================================================
# 유틸
# ============================================================
def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


PORT_OUTWARD_NORMAL_UNIT = normalize(PORT_OUTWARD_NORMAL)
INSERTION_DIRECTION_UNIT = normalize(INSERTION_DIRECTION)


def make_outward_point(center: np.ndarray, distance: float) -> np.ndarray:
    return center + PORT_OUTWARD_NORMAL_UNIT * distance


def find_prim_path_by_name(root_path: str, name: str):
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def get_prim_world_position(prim_path: str):
    if not prim_path:
        return None
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.XformCache()
    mat = cache.GetLocalToWorldTransform(prim)
    t = mat.ExtractTranslation()
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)


def get_prim_world_rotation(prim_path: str):
    if not prim_path:
        return None
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.XformCache()
    mat = cache.GetLocalToWorldTransform(prim)
    return mat.ExtractRotation()


def angle_deg_between(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-9):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return None
    c = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def find_dof_index(robot, dof_name: str):
    if hasattr(robot, "dof_names") and dof_name in robot.dof_names:
        return robot.dof_names.index(dof_name)
    return None


def build_initial_joint_positions(robot, base_positions=None) -> np.ndarray:
    if base_positions is None:
        q = np.zeros(robot.num_dof, dtype=float)
    else:
        q = np.array(base_positions, dtype=float).copy()
        if len(q) != robot.num_dof:
            q = np.zeros(robot.num_dof, dtype=float)

    missing_arm = []
    for joint_name, deg in INITIAL_ARM_JOINT_DEG.items():
        idx = find_dof_index(robot, joint_name)
        if idx is None:
            missing_arm.append(joint_name)
        else:
            q[idx] = np.deg2rad(deg)

    if missing_arm and robot.num_dof >= 6:
        fallback_values = list(INITIAL_ARM_JOINT_DEG.values())
        for i, deg in enumerate(fallback_values):
            q[i] = np.deg2rad(deg)

    for joint_name, value in INITIAL_GRIPPER_JOINTS.items():
        idx = find_dof_index(robot, joint_name)
        if idx is not None:
            q[idx] = value

    return q


def apply_robot_start_state(robot, base_world: np.ndarray, base_orientation: np.ndarray):
    robot.set_world_pose(position=base_world, orientation=base_orientation)
    current = robot.get_joint_positions()
    q0 = build_initial_joint_positions(robot, current)
    robot.set_joint_positions(q0)
    return q0


def initialize_robot(robot, world, base_world: np.ndarray, base_orientation: np.ndarray):
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )
    q0 = apply_robot_start_state(robot, base_world, base_orientation)
    robot.gripper.set_joint_positions(np.array(GRIPPER_OPEN, dtype=float))
    return q0


def step_home_return(robot, target_joint_positions: np.ndarray,
                      alpha: float = HOME_JOINT_SPEED_ALPHA, tol: float = HOME_JOINT_TOLERANCE) -> bool:
    current = robot.get_joint_positions()
    next_joints = current + alpha * (target_joint_positions - current)
    robot.set_joint_positions(next_joints)
    joint_err = np.linalg.norm(next_joints[:6] - target_joint_positions[:6])
    return bool(joint_err < tol)


def set_single_joint(robot, joint_index: int, value: float):
    current = robot.get_joint_positions()
    current[joint_index] = value
    robot.set_joint_positions(current)


# ============================================================
# 카메라 좌표 변환 (단일 벽 부착 카메라를 A/B가 공유)
# ============================================================
def find_camera_prim_path():
    stage = omni.usd.get_context().get_stage()
    for path in CAMERA_PRIM_CANDIDATES:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            return path
    root = stage.GetPrimAtPath(SCENE_SEARCH_ROOT)
    if root.IsValid():
        for prim in Usd.PrimRange(root):
            if prim.GetTypeName() == "Camera" or prim.GetName().lower() in ["camera", "rsd455"]:
                return str(prim.GetPath())
    return None


def camera_ros_point_to_usd_camera_local(point_camera_ros: np.ndarray) -> np.ndarray:
    x, y, z = [float(v) for v in point_camera_ros]
    if CAMERA_POINT_CONVENTION == "ros_optical":
        return np.array([x, -y, -z], dtype=float)
    return np.array([x, y, z], dtype=float)


def transform_camera_point_to_world(point_camera_ros: np.ndarray, camera_prim_path: str):
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(camera_prim_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.XformCache()
    mat = cache.GetLocalToWorldTransform(prim)
    t = mat.ExtractTranslation()
    camera_origin_world = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)

    x_cam = float(point_camera_ros[0])
    y_cam = float(point_camera_ros[1])
    z_cam = float(point_camera_ros[2])

    delta_world = np.array([-x_cam, -z_cam, -y_cam], dtype=float)
    return camera_origin_world + delta_world


def detected_world_point_to_mouth_center(detected_world_point: np.ndarray) -> np.ndarray:
    return detected_world_point - PORT_OUTWARD_NORMAL_UNIT * (FUEL_PORT_DEPTH / 2.0)


def validate_detected_center_world(center_world: np.ndarray, reference_center: np.ndarray,
                                    gate_half_extent: np.ndarray = SEARCH_GATE_HALF_EXTENT):
    if center_world is None or not np.all(np.isfinite(center_world)):
        return False, "non-finite target"
    delta = center_world - reference_center
    if np.any(np.abs(delta) > gate_half_extent):
        return False, f"outside gate: delta={np.round(delta, 3)}"
    return True, f"inside gate: delta={np.round(delta, 3)}"


# ============================================================
# 도어 힌지 geometry: Revolution Joint의 축/피벗을 USD에서 읽어온다
# ============================================================
def find_revolute_joint_for_body(body_prim_path: str):
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(SCENE_SEARCH_ROOT)
    if not root.IsValid() or not body_prim_path:
        return None
    for prim in Usd.PrimRange(root):
        joint = UsdPhysics.RevoluteJoint(prim)
        if not joint:
            continue
        targets = list(joint.GetBody0Rel().GetTargets()) + list(joint.GetBody1Rel().GetTargets())
        if any(str(t) == body_prim_path for t in targets):
            return prim
    return None


def get_joint_world_axis_and_pivot(joint_prim):
    """RevoluteJoint의 axis/localPos0를 body0 world transform 기준으로 변환한다."""
    joint = UsdPhysics.RevoluteJoint(joint_prim)
    axis_token = joint.GetAxisAttr().Get() or "X"
    local_axis = {
        "X": Gf.Vec3d(1, 0, 0),
        "Y": Gf.Vec3d(0, 1, 0),
        "Z": Gf.Vec3d(0, 0, 1),
    }.get(str(axis_token), Gf.Vec3d(1, 0, 0))

    body0_targets = joint.GetBody0Rel().GetTargets()
    stage = omni.usd.get_context().get_stage()
    if body0_targets:
        body0_prim = stage.GetPrimAtPath(body0_targets[0])
        cache = UsdGeom.XformCache()
        body0_world = cache.GetLocalToWorldTransform(body0_prim)
    else:
        body0_world = Gf.Matrix4d(1.0)

    local_pos0 = joint.GetLocalPos0Attr().Get() or Gf.Vec3f(0, 0, 0)
    local_rot0 = joint.GetLocalRot0Attr().Get()

    if local_rot0 is not None:
        rot = Gf.Rotation(Gf.Quatd(local_rot0.GetReal(), Gf.Vec3d(local_rot0.GetImaginary())))
        axis_in_body0 = rot.TransformDir(local_axis)
    else:
        axis_in_body0 = local_axis

    axis_world_gf = body0_world.TransformDir(axis_in_body0)
    axis_world = normalize(np.array([axis_world_gf[0], axis_world_gf[1], axis_world_gf[2]], dtype=float))

    pivot_world_gf = body0_world.Transform(Gf.Vec3d(local_pos0))
    pivot_world = np.array([pivot_world_gf[0], pivot_world_gf[1], pivot_world_gf[2]], dtype=float)
    return axis_world, pivot_world


def rotate_point_around_axis(point: np.ndarray, pivot: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    rot = Gf.Rotation(Gf.Vec3d(*[float(a) for a in axis]), float(angle_deg))
    rel = Gf.Vec3d(*[float(p) for p in (point - pivot)])
    rotated_rel = rot.TransformDir(rel)
    return pivot + np.array([rotated_rel[0], rotated_rel[1], rotated_rel[2]], dtype=float)


def signed_angle_about_axis_deg(rest_rotation, current_rotation, axis_world: np.ndarray) -> float:
    if rest_rotation is None or current_rotation is None:
        return 0.0
    rel = current_rotation * rest_rotation.GetInverse()
    angle = float(rel.GetAngle())
    axis = rel.GetAxis()
    axis_np = np.array([axis[0], axis[1], axis[2]], dtype=float)
    if np.linalg.norm(axis_np) < 1e-6:
        return 0.0
    sign = 1.0 if np.dot(axis_np, axis_world) >= 0.0 else -1.0
    return DOOR_ANGLE_SIGN * sign * angle


# ============================================================
# Stage / WaypointSequence: 속도 제한 RMPFlow 목표점 state machine
# ============================================================
@dataclass
class FuelStage:
    name: str
    target_position: "np.ndarray | None"
    hold_steps: int = 0
    tolerance: float = POSITION_TOLERANCE
    max_steps: int = MAX_STEPS_PER_STAGE
    speed: float = DEFAULT_TARGET_SPEED
    use_orientation: bool = True
    target_door_angle: "float | None" = None
    door_angle_tolerance: float = DOOR_ANGLE_TOLERANCE_DEG


class WaypointSequence:
    """link_6 목표를 속도 제한된 중간 목표로 천천히 이동시키는 범용 stage state machine."""

    def __init__(self, stages: list, stage_log: dict | None = None):
        self.stages = stages
        self.stage_log = stage_log or {}
        self.index = 0
        self.stage_step = 0
        self.hold_count = 0
        self.done = False
        self.command_target = None

    @property
    def current(self) -> FuelStage:
        return self.stages[self.index]

    def reset(self):
        self.index = 0
        self.stage_step = 0
        self.hold_count = 0
        self.done = False
        self.command_target = None

    def update(self, ee_position: np.ndarray, extra_condition_ok: bool = True) -> bool:
        if self.done:
            return True
        stage = self.current
        if stage.target_position is None:
            return False

        self.stage_step += 1
        err = np.linalg.norm(stage.target_position - ee_position)
        reached = (err < stage.tolerance) and extra_condition_ok
        timed_out = self.stage_step >= stage.max_steps

        if reached:
            self.hold_count += 1
        else:
            self.hold_count = 0

        if (reached and self.hold_count >= stage.hold_steps) or timed_out:
            label = self.stage_log.get(stage.name, stage.name)
            if timed_out and not reached:
                print(f"\n[⚠️ 타임아웃] {label} 남은거리={err:.3f}m")
            else:
                print(f"\n[✅ {label}] 위치={np.round(ee_position, 3)}")
            self.index += 1
            self.stage_step = 0
            self.hold_count = 0
            self.command_target = None
            if self.index >= len(self.stages):
                self.done = True
                return True
        return self.done

    def get_command_target(self, ee_position: np.ndarray):
        if self.done:
            return None
        stage = self.current
        if stage.target_position is None:
            return None
        if self.command_target is None:
            self.command_target = np.array(ee_position, dtype=float)

        delta = stage.target_position - self.command_target
        dist = np.linalg.norm(delta)
        max_step = max(stage.speed * PHYSICS_DT, 1e-5)
        if dist <= max_step:
            self.command_target = np.array(stage.target_position, dtype=float)
        else:
            self.command_target = self.command_target + delta / dist * max_step
        return self.command_target

    def debug_string(self, ee_position: np.ndarray) -> str:
        if self.done:
            return "[DONE]"
        stage = self.current
        if stage.target_position is None:
            return f"[stage={self.index}:{stage.name}] hold={self.hold_count}/{stage.hold_steps}"
        err = np.linalg.norm(stage.target_position - ee_position)
        return (
            f"[stage={self.index}:{stage.name}] target={np.round(stage.target_position, 3)} "
            f"ee={np.round(ee_position, 3)} err={err:.4f} speed={stage.speed:.3f} "
            f"hold={self.hold_count}/{stage.hold_steps}"
        )


def single_stage_sequence(name: str, target: np.ndarray, **kwargs) -> WaypointSequence:
    return WaypointSequence([FuelStage(name, target, **kwargs)])


# ============================================================
# A 로봇: 기존 주유 시퀀스 (기존 FuelPortSequence 로직 그대로, 색만 green)
# ============================================================
def build_fuel_port_sequence(fuel_port_center: np.ndarray) -> WaypointSequence:
    outward = PORT_OUTWARD_NORMAL_UNIT
    insertion = INSERTION_DIRECTION_UNIT

    tip_far  = fuel_port_center + outward * FAR_DISTANCE
    tip_mid  = fuel_port_center + outward * MID_DISTANCE
    tip_near = fuel_port_center + outward * NEAR_DISTANCE
    tip_insert = fuel_port_center + insertion * INSERT_DISTANCE

    control_offset = outward * VIRTUAL_NOZZLE_LENGTH + np.array([0.0, 0.0, VIRTUAL_NOZZLE_Z_OFFSET])
    approach_far  = tip_far + control_offset
    approach_mid  = tip_mid + control_offset
    approach_near = tip_near + control_offset
    insert_target = tip_insert + control_offset

    stages = [
        FuelStage("01_axis_far_start", approach_far, tolerance=0.030, speed=DEFAULT_TARGET_SPEED),
        FuelStage("02_axis_mid", approach_mid, tolerance=0.028, speed=DEFAULT_TARGET_SPEED),
        FuelStage("03_axis_near_stop", approach_near, hold_steps=80, tolerance=0.022, speed=NEAR_TARGET_SPEED),
        FuelStage("04_insert_into_cylinder", insert_target, hold_steps=180, tolerance=0.018, speed=INSERT_TARGET_SPEED),
        FuelStage("05_retreat_near", approach_near, speed=RETREAT_TARGET_SPEED),
        FuelStage("06_retreat_mid", approach_mid, speed=RETREAT_TARGET_SPEED),
        FuelStage("07_retreat_far", approach_far, hold_steps=30, tolerance=0.035, speed=RETREAT_TARGET_SPEED),
        FuelStage("08_return_home", None),
    ]
    stage_log = {
        "01_axis_far_start": "A 접근 시작",
        "02_axis_mid": "A 중간 지점",
        "03_axis_near_stop": "A 근처 대기",
        "04_insert_into_cylinder": "A 삽입 완료",
        "05_retreat_near": "A 후퇴 시작",
        "06_retreat_mid": "A 중간 후퇴",
        "07_retreat_far": "A 후퇴 완료",
    }
    return WaypointSequence(stages, stage_log)


# ============================================================
# B 로봇: 커버 push 시퀀스 / 마개 접근-복원 시퀀스
# ============================================================
def build_cover_sequence(direction: str, door_reference_point: np.ndarray, pivot: np.ndarray, axis: np.ndarray) -> WaypointSequence:
    """direction: 'open' (30->80->130) 또는 'close' (130->80->30)."""
    def p(angle_deg):
        return rotate_point_around_axis(door_reference_point, pivot, axis, angle_deg - COVER_START_DEG)

    if direction == "open":
        stages = [
            FuelStage("B2_01_approach_30", p(COVER_START_DEG), tolerance=0.030, speed=DEFAULT_TARGET_SPEED,
                       target_door_angle=COVER_START_DEG),
            FuelStage("B2_02_push_80", p(COVER_MID_DEG), hold_steps=40, tolerance=0.025, speed=NEAR_TARGET_SPEED,
                       target_door_angle=COVER_MID_DEG),
            FuelStage("B2_03_push_130", p(COVER_OPEN_DEG), hold_steps=60, tolerance=0.025, speed=NEAR_TARGET_SPEED,
                       target_door_angle=COVER_OPEN_DEG),
        ]
        stage_log = {
            "B2_01_approach_30": "B 커버 접촉",
            "B2_02_push_80": "B 커버 80도",
            "B2_03_push_130": "B 커버 130도 완전열림",
        }
    else:
        stages = [
            FuelStage("B10_01_reengage_130", p(COVER_OPEN_DEG), tolerance=0.030, speed=DEFAULT_TARGET_SPEED,
                       target_door_angle=COVER_OPEN_DEG),
            FuelStage("B10_02_push_80", p(COVER_MID_DEG), hold_steps=40, tolerance=0.025, speed=NEAR_TARGET_SPEED,
                       target_door_angle=COVER_MID_DEG),
            FuelStage("B10_03_push_30", p(COVER_START_DEG), hold_steps=60, tolerance=0.025, speed=NEAR_TARGET_SPEED,
                       target_door_angle=COVER_START_DEG),
        ]
        stage_log = {
            "B10_01_reengage_130": "B 커버 재접촉(130)",
            "B10_02_push_80": "B 커버 80도(닫는중)",
            "B10_03_push_30": "B 커버 30도 닫힘완료",
        }
    return WaypointSequence(stages, stage_log)


def build_cap_approach_sequence(cap_center: np.ndarray) -> WaypointSequence:
    stages = [
        FuelStage("B5_01_far", make_outward_point(cap_center, FAR_DISTANCE), tolerance=0.030, speed=DEFAULT_TARGET_SPEED),
        FuelStage("B5_02_near", make_outward_point(cap_center, NEAR_DISTANCE), hold_steps=60, tolerance=0.022, speed=NEAR_TARGET_SPEED),
        FuelStage("B5_03_grasp", cap_center, hold_steps=80, tolerance=0.018, speed=INSERT_TARGET_SPEED),
    ]
    stage_log = {
        "B5_01_far": "B 마개 접근(far)",
        "B5_02_near": "B 마개 접근(near)",
        "B5_03_grasp": "B 마개 grasp 위치 도착",
    }
    return WaypointSequence(stages, stage_log)


def build_cap_extract_sequence(cap_center: np.ndarray) -> WaypointSequence:
    return single_stage_sequence(
        "B6_extract", make_outward_point(cap_center, NEAR_DISTANCE),
        hold_steps=30, tolerance=0.025, speed=RETREAT_TARGET_SPEED,
    )


def build_cap_restore_sequence(hole_center: np.ndarray) -> WaypointSequence:
    stages = [
        FuelStage("B9_01_far", make_outward_point(hole_center, FAR_DISTANCE), tolerance=0.030, speed=DEFAULT_TARGET_SPEED),
        FuelStage("B9_02_near", make_outward_point(hole_center, NEAR_DISTANCE), hold_steps=60, tolerance=0.022, speed=NEAR_TARGET_SPEED),
        FuelStage("B9_03_insert", hole_center, hold_steps=80, tolerance=0.018, speed=INSERT_TARGET_SPEED),
    ]
    stage_log = {
        "B9_01_far": "B 주유구 접근(far)",
        "B9_02_near": "B 주유구 접근(near)",
        "B9_03_insert": "B 마개 삽입 위치 도착",
    }
    return WaypointSequence(stages, stage_log)


# ============================================================
# ROS2 bridge: multi_color_detector.py 및 A/B 동기화 토픽을 한 노드에서 처리
# ============================================================
class MultiRobotRosBridge(Node):
    def __init__(self):
        super().__init__("multi_robot_oiling_ros_bridge")
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        # mode_switch/robot_a_done/robot_b_done은 한 번만 발행되는 이벤트 플래그라,
        # 구독자가 그 시점에 늦게 join해도 마지막 값을 받을 수 있도록 latch 시킨다.
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.latest_pose: "PoseStamped | None" = None
        self.pose_count = 0
        self.target_locked = False
        self.lock_count = 0
        self.robot_a_done = False
        self.robot_a_done_count = 0
        self.robot_b_done = False
        self.robot_b_done_count = 0

        self.pose_sub = self.create_subscription(PoseStamped, TOPIC_COLOR_POSE, self._pose_cb, sensor_qos)
        self.lock_sub = self.create_subscription(Bool, TOPIC_COLOR_LOCK, self._lock_cb, sensor_qos)
        self.robot_a_done_sub = self.create_subscription(Bool, TOPIC_ROBOT_A_DONE, self._robot_a_done_cb, latched_qos)
        self.robot_b_done_sub = self.create_subscription(Bool, TOPIC_ROBOT_B_DONE, self._robot_b_done_cb, latched_qos)

        self.mode_switch_pub = self.create_publisher(String, TOPIC_MODE_SWITCH, latched_qos)
        self.robot_a_done_pub = self.create_publisher(Bool, TOPIC_ROBOT_A_DONE, latched_qos)
        self.robot_b_done_pub = self.create_publisher(Bool, TOPIC_ROBOT_B_DONE, latched_qos)

        self.get_logger().info("MultiRobotRosBridge started")

    def _pose_cb(self, msg: PoseStamped):
        self.latest_pose = msg
        self.pose_count += 1

    def _lock_cb(self, msg: Bool):
        self.target_locked = bool(msg.data)
        self.lock_count += 1

    def _robot_a_done_cb(self, msg: Bool):
        self.robot_a_done = bool(msg.data)
        self.robot_a_done_count += 1

    def _robot_b_done_cb(self, msg: Bool):
        self.robot_b_done = bool(msg.data)
        self.robot_b_done_count += 1

    def get_pose_if_ready(self):
        if self.latest_pose is None:
            return None
        if REQUIRE_TARGET_LOCK and not self.target_locked:
            return None
        return self.latest_pose

    def publish_mode_switch(self, mode: str):
        self.mode_switch_pub.publish(String(data=mode))
        self.get_logger().info(f"mode_switch -> {mode}")

    def publish_robot_a_done(self, flag: bool):
        self.robot_a_done_pub.publish(Bool(data=bool(flag)))

    def publish_robot_b_done(self, flag: bool):
        self.robot_b_done_pub.publish(Bool(data=bool(flag)))


class StableTargetLockAcquirer:
    """노란/파란/초록 색 모드 공통: N개 샘플이 표준편차 이내로 모이면 평균 world 좌표를 반환한다."""

    def __init__(self, ros_bridge: MultiRobotRosBridge, camera_prim_path: "str | None",
                 reference_center: np.ndarray, apply_mouth_offset: bool = False,
                 lock_z_to_reference: bool = True,
                 required_samples: int = CONTROLLER_REQUIRED_LOCK_SAMPLES,
                 std_tolerance: float = CONTROLLER_WORLD_STD_TOLERANCE,
                 gate_half_extent: np.ndarray = SEARCH_GATE_HALF_EXTENT):
        self.ros_bridge = ros_bridge
        self.camera_prim_path = camera_prim_path
        self.reference_center = reference_center
        self.apply_mouth_offset = apply_mouth_offset
        self.lock_z_to_reference = lock_z_to_reference
        self.required_samples = required_samples
        self.std_tolerance = std_tolerance
        self.gate_half_extent = gate_half_extent
        self.samples: list = []
        self.last_sampled_pose_count = -1

    def reset(self):
        self.samples = []
        self.last_sampled_pose_count = -1

    def update(self):
        """매 tick 호출. 안정화된 평균 world 좌표가 나오면 반환, 아니면 None.

        target_locked가 잠깐 False로 흔들려도(디텍터는 publish_hz로만 lock을 발행하므로
        그 순간의 깜빡임이 그대로 들어온다) 누적된 샘플을 통째로 비우지 않는다.
        get_pose_if_ready()가 이미 unlocked 상태에서는 새 샘플 추가를 막아주므로,
        여기서 추가로 버퍼를 비우면 5개를 채우기 전에 계속 리셋되어 영원히 lock이 안 된다.
        """
        pose_msg = self.ros_bridge.get_pose_if_ready()
        if pose_msg is None or self.ros_bridge.pose_count == self.last_sampled_pose_count:
            return None
        self.last_sampled_pose_count = self.ros_bridge.pose_count

        if self.camera_prim_path is None:
            print("[LOCK] camera_prim_path가 None이라 pose를 world 좌표로 변환할 수 없음 "
                  "(find_camera_prim_path()가 카메라 prim을 못 찾음 - CAMERA_PRIM_CANDIDATES 확인)")
            return None

        p_cam = np.array([
            pose_msg.pose.position.x,
            pose_msg.pose.position.y,
            pose_msg.pose.position.z,
        ], dtype=float)
        detected_world_point = transform_camera_point_to_world(p_cam, self.camera_prim_path)
        if detected_world_point is None:
            print(f"[LOCK] transform_camera_point_to_world() 실패 (camera_prim_path={self.camera_prim_path} 가 invalid)")
            return None

        candidate = (
            detected_world_point_to_mouth_center(detected_world_point)
            if self.apply_mouth_offset else detected_world_point
        )
        if self.lock_z_to_reference:
            candidate[2] = self.reference_center[2]

        valid, reason = validate_detected_center_world(candidate, self.reference_center, self.gate_half_extent)
        if valid:
            self.samples.append(candidate)
            self.samples = self.samples[-self.required_samples:]
        else:
            self.samples = []
            print(f"[LOCK REJECT] p_cam={np.round(p_cam, 3)} world_pt={np.round(detected_world_point, 3)} "
                  f"candidate={np.round(candidate, 3)} ref={np.round(self.reference_center, 3)} reason={reason}")

        if len(self.samples) >= self.required_samples:
            arr = np.array(self.samples, dtype=float)
            mean = arr.mean(axis=0)
            std_norm = float(np.linalg.norm(arr.std(axis=0)))
            if self.lock_z_to_reference:
                mean[2] = self.reference_center[2]
            if std_norm <= self.std_tolerance:
                return mean
        return None


# ============================================================
# Task: USD 로드, 두 로봇 등록, fuel_door/cap/hole 및 힌지 joint 탐색
# ============================================================
class MultiRobotOilingTask(BaseTask):
    def __init__(self, name):
        super().__init__(name=name, offset=None)

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._load_usd()
        self._discover_robot_links()
        self._setup_physics()
        self._register_robots(scene)
        self._discover_fuel_objects(scene)
        print("\n  [완료] multi-robot oiling 씬 구성 성공!\n")

    def _load_usd(self):
        print("\n" + "=" * 60)
        print("[1.LOAD] USD 로드")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()
        world_prim = stage.GetPrimAtPath("/World")
        if not world_prim.IsValid():
            world_prim = UsdGeom.Xform.Define(stage, "/World").GetPrim()
        world_prim.GetReferences().AddReference(USD_PATH)
        for _ in range(15):
            simulation_app.update()
        print(f"  [OK] {USD_PATH}")
        print(f"  [NOTE] m0609_A={ROBOT_A_PRIM_PATH}, m0609_B={ROBOT_B_PRIM_PATH}가 USD에 이미 있다고 가정")

    def _discover_robot_links(self):
        print("\n" + "=" * 60)
        print("[2.DISCOVER] 로봇 A/B 링크 경로 탐색")
        print("=" * 60)
        self.ee_path_a = find_prim_path_by_name(ROBOT_A_PRIM_PATH, EE_LINK_NAME)
        self.ee_path_b = find_prim_path_by_name(ROBOT_B_PRIM_PATH, EE_LINK_NAME)
        if self.ee_path_a is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found under {ROBOT_A_PRIM_PATH}")
        if self.ee_path_b is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found under {ROBOT_B_PRIM_PATH}")
        print(f"  A EE = {self.ee_path_a}")
        print(f"  B EE = {self.ee_path_b}")

    def _setup_physics(self):
        print("\n" + "=" * 60)
        print("[3.PHYSICS] 로봇 A/B drive 설정")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()
        drive_count = 0
        for root_path in (ROBOT_A_PRIM_PATH, ROBOT_B_PRIM_PATH):
            root = stage.GetPrimAtPath(root_path)
            if not root.IsValid():
                raise RuntimeError(f"Robot prim not found: {root_path}")
            for prim in Usd.PrimRange(root):
                for dt in ["angular", "linear"]:
                    drive = UsdPhysics.DriveAPI.Get(prim, dt)
                    if drive:
                        drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                        drive.GetDampingAttr().Set(DRIVE_DAMPING)
                        drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                        drive_count += 1
        print(f"  [OK] drive updated: {drive_count}")

    def _register_robots(self, scene):
        print("\n" + "=" * 60)
        print("[4.REGISTER] SingleManipulator A/B 등록")
        print("=" * 60)
        gripper_a = ParallelGripper(
            end_effector_prim_path=self.ee_path_a,
            joint_prim_names=GRIPPER_JOINTS,
            joint_opened_positions=np.array(GRIPPER_OPEN),
            joint_closed_positions=np.array(GRIPPER_CLOSE),
            action_deltas=np.array(GRIPPER_DELTA),
        )
        self.robot_a = scene.add(
            SingleManipulator(
                prim_path=ROBOT_A_PRIM_PATH,
                name="m0609_A",
                end_effector_prim_path=self.ee_path_a,
                gripper=gripper_a,
            )
        )
        gripper_b = ParallelGripper(
            end_effector_prim_path=self.ee_path_b,
            joint_prim_names=GRIPPER_JOINTS,
            joint_opened_positions=np.array(GRIPPER_OPEN),
            joint_closed_positions=np.array(GRIPPER_CLOSE),
            action_deltas=np.array(GRIPPER_DELTA),
        )
        self.robot_b = scene.add(
            SingleManipulator(
                prim_path=ROBOT_B_PRIM_PATH,
                name="m0609_B",
                end_effector_prim_path=self.ee_path_b,
                gripper=gripper_b,
            )
        )
        print(f"  [OK] m0609_A = {ROBOT_A_PRIM_PATH}")
        print(f"  [OK] m0609_B = {ROBOT_B_PRIM_PATH}")

    def _resolve_world_position(self, prim_path, fallback_constant, label):
        if prim_path is not None:
            pos = get_prim_world_position(prim_path)
            if pos is not None:
                diff = pos - fallback_constant
                print(f"  [OK] {label} 실제 world 위치 = {np.round(pos, 4)} "
                      f"(하드코딩 상수와 차이 = {np.round(diff, 4)}, |diff|={np.linalg.norm(diff):.4f}m)")
                return pos
        print(f"  [WARN] {label} prim 위치를 못 읽음 -> 하드코딩 상수 {np.round(fallback_constant, 4)} 로 폴백")
        return fallback_constant.copy()

    def _discover_fuel_objects(self, scene):
        print("\n" + "=" * 60)
        print("[5.SCENE] fuel_door / fuel_cap / fuel_port_hole 탐색")
        print("=" * 60)
        self.door_prim_path = find_prim_path_by_name(SCENE_SEARCH_ROOT, FUEL_DOOR_PRIM_NAME)
        self.cap_prim_path = find_prim_path_by_name(SCENE_SEARCH_ROOT, FUEL_CAP_PRIM_NAME)
        self.hole_prim_path = find_prim_path_by_name(SCENE_SEARCH_ROOT, FUEL_PORT_HOLE_PRIM_NAME)
        print(f"  fuel_door = {self.door_prim_path}")
        print(f"  fuel_cap  = {self.cap_prim_path}")
        print(f"  fuel_port_hole = {self.hole_prim_path}")

        # 게이트 판정/실제 모션 목표는 프롬프트의 하드코딩 좌표가 아니라 USD 씬에서 직접 읽은
        # 실제 world 위치를 우선 사용한다. 하드코딩 값은 해당 prim을 못 찾았을 때만 폴백으로 쓴다.
        self.door_world_position = self._resolve_world_position(self.door_prim_path, FUEL_DOOR_CENTER, "fuel_door")
        self.cap_world_position = self._resolve_world_position(self.cap_prim_path, FUEL_CAP_CENTER, "fuel_cap")
        self.hole_world_position = self._resolve_world_position(self.hole_prim_path, FUEL_PORT_HOLE_CENTER, "fuel_port_hole")

        self.door_joint_prim = None
        self.door_axis_world = np.array([0.0, 0.0, 1.0], dtype=float)
        self.door_pivot_world = self.door_world_position.copy()
        if self.door_prim_path is not None:
            self.door_joint_prim = find_revolute_joint_for_body(self.door_prim_path)
            if self.door_joint_prim is not None:
                self.door_axis_world, self.door_pivot_world = get_joint_world_axis_and_pivot(self.door_joint_prim)
                print(f"  [OK] door hinge axis(world)={np.round(self.door_axis_world, 3)} "
                      f"pivot(world)={np.round(self.door_pivot_world, 3)}")
            else:
                print("  [WARN] fuel_door에 연결된 RevoluteJoint를 찾지 못함. "
                      "기본 축([0,0,1])과 FUEL_DOOR_CENTER를 피벗으로 가정함 - 실제 동작 전 검증 필요.")

        # 리셋 시 door의 "기준 회전"을 캐시해서 이후 상대 각도를 측정한다 (COVER_START_DEG로 가정).
        self.door_rest_rotation = get_prim_world_rotation(self.door_prim_path)

        markers = [
            ("fuel_marker_door", self.door_world_position, np.array([1.0, 1.0, 0.0])),
            ("fuel_marker_cap", self.cap_world_position, np.array([0.0, 0.5, 1.0])),
            ("fuel_marker_hole", self.hole_world_position, np.array([0.0, 1.0, 0.3])),
        ]
        for marker_name, pos, color in markers:
            scene.add(
                VisualCuboid(
                    prim_path=f"/World/{marker_name}",
                    name=marker_name,
                    position=pos,
                    scale=np.array([0.02, 0.02, 0.02]),
                    color=color,
                )
            )

    def current_door_angle_deg(self) -> float:
        current_rotation = get_prim_world_rotation(self.door_prim_path)
        delta = signed_angle_about_axis_deg(self.door_rest_rotation, current_rotation, self.door_axis_world)
        return COVER_START_DEG + delta

    def post_reset(self):
        self.robot_a.gripper.set_joint_positions(self.robot_a.gripper.joint_opened_positions)
        self.robot_b.gripper.set_joint_positions(self.robot_b.gripper.joint_opened_positions)
        self.door_rest_rotation = get_prim_world_rotation(self.door_prim_path)


# ============================================================
# Robot A runner: WAIT_LOCK(green) -> RUN_SEQUENCE(기존 주유 로직)
# ============================================================
class RobotARunner:
    def __init__(self, robot, controller, ros_bridge: MultiRobotRosBridge, camera_prim_path,
                 task: "MultiRobotOilingTask"):
        self.robot = robot
        self.controller = controller
        self.ros_bridge = ros_bridge
        self.camera_prim_path = camera_prim_path
        self.task = task
        self.run_state = "IDLE_WAIT_B"
        self.lock_acquirer = None
        self.sequence = None
        self.locked_target_orientation = None
        self.task_done = False
        self.wait_steps = 0
        self._b_done_count_at_reset = 0

    def on_play_reset(self):
        _, ee_ori = self.robot.end_effector.get_world_pose()
        self.locked_target_orientation = ee_ori.copy()
        self.run_state = "IDLE_WAIT_B"
        self.lock_acquirer = None
        self.sequence = None
        self.task_done = False
        self.wait_steps = 0
        self._b_done_count_at_reset = self.ros_bridge.robot_b_done_count
        print("[A] RESET -> IDLE_WAIT_B (robot_b/done 대기)")

    def _robot_b_done_received(self) -> bool:
        return (
            self.ros_bridge.robot_b_done_count > self._b_done_count_at_reset
            and self.ros_bridge.robot_b_done
        )

    def tick(self, step_count: int):
        if self.task_done:
            return
        ee_pos, _ = self.robot.end_effector.get_world_pose()

        if self.run_state == "IDLE_WAIT_B":
            if self._robot_b_done_received():
                self.ros_bridge.publish_mode_switch("green")
                self.lock_acquirer = StableTargetLockAcquirer(
                    self.ros_bridge, self.camera_prim_path, self.task.hole_world_position, apply_mouth_offset=True,
                )
                self.wait_steps = 0
                self.run_state = "WAIT_LOCK_GREEN"
                print("\n[A-8] robot_b/done 수신 -> mode_switch=green, 주유구 lock 대기 시작\n")
            return

        if self.run_state == "WAIT_LOCK_GREEN":
            self.wait_steps += 1
            mean = self.lock_acquirer.update()
            if mean is not None:
                self.sequence = build_fuel_port_sequence(mean)
                self.run_state = "RUN_SEQUENCE"
                print(f"\n[A] green lock 완료. fuel_port_hole center={np.round(mean, 4)}\n")
                return
            if self.wait_steps >= WAIT_LOCK_TIMEOUT_STEPS:
                print("\n[A] WAIT_LOCK_GREEN timeout -> 재시도\n")
                self.wait_steps = 0
                self.lock_acquirer.reset()
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print(f"[A][WAIT_LOCK_GREEN] locked={self.ros_bridge.target_locked} "
                      f"pose_count={self.ros_bridge.pose_count} lock_count={self.ros_bridge.lock_count} "
                      f"samples={len(self.lock_acquirer.samples)}/{CONTROLLER_REQUIRED_LOCK_SAMPLES}")
            return

        if self.run_state == "RUN_SEQUENCE":
            stage = self.sequence.current
            if stage.name == "08_return_home":
                target_joints = build_initial_joint_positions(self.robot, self.robot.get_joint_positions())
                reached = step_home_return(self.robot, target_joints)
                self.sequence.hold_count = self.sequence.hold_count + 1 if reached else 0
                if step_count % PRINT_EVERY_N_STEPS == 0:
                    print(f"[A][08_return_home] hold={self.sequence.hold_count}/{HOME_HOLD_STEPS}")
                if self.sequence.hold_count >= HOME_HOLD_STEPS:
                    self.task_done = True
                    self.ros_bridge.publish_robot_a_done(True)
                    print("\n[A] 주유 완료 -> robot_a/done = True 발행\n")
                return

            done = self.sequence.update(ee_pos)
            if done:
                return
            stage = self.sequence.current
            cmd = self.sequence.get_command_target(ee_pos)
            if cmd is not None:
                if stage.use_orientation and USE_TARGET_ORIENTATION:
                    actions = self.controller.forward(
                        target_end_effector_position=cmd,
                        target_end_effector_orientation=self.locked_target_orientation,
                    )
                else:
                    actions = self.controller.forward(target_end_effector_position=cmd)
                self.robot.apply_action(actions)
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print("[A] " + self.sequence.debug_string(ee_pos))


# ============================================================
# Robot B runner: 11단계 상태머신
# WAIT_YELLOW -> OPEN_COVER -> RETURN_MID -> WAIT_BLUE -> MOVE_TO_CAP ->
# GRIP_UNSCREW -> RETURN_HOME_WITH_CAP -> WAIT_ROBOT_A -> RESTORE_CAP ->
# CLOSE_COVER -> FINAL_HOME
# ============================================================
class RobotBRunner:
    def __init__(self, robot, controller, ros_bridge: MultiRobotRosBridge, camera_prim_path, task: MultiRobotOilingTask):
        self.robot = robot
        self.controller = controller
        self.ros_bridge = ros_bridge
        self.camera_prim_path = camera_prim_path
        self.task = task
        self.run_state = "WAIT_YELLOW"
        self.sub_phase = None
        self.lock_acquirer = None
        self.sequence = None
        self.locked_target_orientation = None
        self.locked_door_center = task.door_world_position.copy()
        self.locked_cap_center = task.cap_world_position.copy()
        self.task_done = False
        self.wait_steps = 0
        self.joint6_index = None
        self.joint6_start_rad = 0.0
        self.joint6_progress_deg = 0.0
        self.gripper_hold_count = 0
        self._a_done_count_at_reset = 0

    def on_play_reset(self):
        _, ee_ori = self.robot.end_effector.get_world_pose()
        self.locked_target_orientation = ee_ori.copy()
        self.run_state = "WAIT_YELLOW"
        self.sub_phase = None
        self.lock_acquirer = StableTargetLockAcquirer(
            self.ros_bridge, self.camera_prim_path, self.task.door_world_position, apply_mouth_offset=False,
        )
        self.sequence = None
        self.task_done = False
        self.wait_steps = 0
        self.gripper_hold_count = 0
        self.joint6_index = find_dof_index(self.robot, "joint_6")
        self._a_done_count_at_reset = self.ros_bridge.robot_a_done_count
        self.ros_bridge.publish_mode_switch("yellow")
        print("[B] RESET -> WAIT_YELLOW (mode_switch=yellow 발행)")

    def _robot_a_done_received(self) -> bool:
        return (
            self.ros_bridge.robot_a_done_count > self._a_done_count_at_reset
            and self.ros_bridge.robot_a_done
        )

    def _hold_gripper(self, closed: bool):
        target = GRIPPER_CLOSE if closed else GRIPPER_OPEN
        self.robot.gripper.set_joint_positions(np.array(target, dtype=float))

    def _drive_sequence(self, ee_pos, extra_condition_ok=True):
        """현재 self.sequence를 한 스텝 진행. 완료 여부를 반환."""
        done = self.sequence.update(ee_pos, extra_condition_ok=extra_condition_ok)
        if done:
            return True
        stage = self.sequence.current
        cmd = self.sequence.get_command_target(ee_pos)
        if cmd is not None:
            if stage.use_orientation and USE_TARGET_ORIENTATION:
                actions = self.controller.forward(
                    target_end_effector_position=cmd,
                    target_end_effector_orientation=self.locked_target_orientation,
                )
            else:
                actions = self.controller.forward(target_end_effector_position=cmd)
            self.robot.apply_action(actions)
        return False

    def _door_angle_ok(self) -> bool:
        stage = self.sequence.current
        if stage.target_door_angle is None:
            return True
        current_angle = self.task.current_door_angle_deg()
        return abs(current_angle - stage.target_door_angle) <= stage.door_angle_tolerance

    def tick(self, step_count: int):
        if self.task_done:
            return
        ee_pos, _ = self.robot.end_effector.get_world_pose()

        # ---------------- B-1: WAIT_YELLOW ----------------
        if self.run_state == "WAIT_YELLOW":
            self.wait_steps += 1
            mean = self.lock_acquirer.update()
            if mean is not None:
                self.locked_door_center = mean
                self.sequence = build_cover_sequence(
                    "open", self.locked_door_center, self.task.door_pivot_world, self.task.door_axis_world,
                )
                self.run_state = "OPEN_COVER"
                print(f"\n[B-1] yellow lock 완료. fuel_door center={np.round(mean, 4)}\n")
                return
            if self.wait_steps >= WAIT_LOCK_TIMEOUT_STEPS:
                print("\n[B-1] WAIT_YELLOW timeout -> 재시도\n")
                self.wait_steps = 0
                self.lock_acquirer.reset()
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print(f"[B][WAIT_YELLOW] locked={self.ros_bridge.target_locked} "
                      f"pose_count={self.ros_bridge.pose_count} lock_count={self.ros_bridge.lock_count} "
                      f"samples={len(self.lock_acquirer.samples)}/{CONTROLLER_REQUIRED_LOCK_SAMPLES}")
            return

        # ---------------- B-2: OPEN_COVER ----------------
        if self.run_state == "OPEN_COVER":
            done = self._drive_sequence(ee_pos, extra_condition_ok=self._door_angle_ok())
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print(f"[B][OPEN_COVER] door_angle={self.task.current_door_angle_deg():.1f} "
                      + self.sequence.debug_string(ee_pos))
            if done:
                mid_point = rotate_point_around_axis(
                    self.locked_door_center, self.task.door_pivot_world, self.task.door_axis_world,
                    COVER_MID_DEG - COVER_START_DEG,
                )
                self.sequence = single_stage_sequence(
                    "B3_return_mid", mid_point, hold_steps=20, tolerance=0.035, speed=RETREAT_TARGET_SPEED,
                )
                self.run_state = "RETURN_MID"
                print("\n[B-2] 커버 완전열림(130도) 완료 -> RETURN_MID\n")
            return

        # ---------------- B-3: RETURN_MID ----------------
        if self.run_state == "RETURN_MID":
            done = self._drive_sequence(ee_pos)
            if done:
                self.lock_acquirer = StableTargetLockAcquirer(
                    self.ros_bridge, self.camera_prim_path, self.task.cap_world_position, apply_mouth_offset=False,
                )
                self.wait_steps = 0
                self.ros_bridge.publish_mode_switch("blue")
                self.run_state = "WAIT_BLUE"
                print("\n[B-3] 중간 지점 복귀 완료 -> mode_switch=blue, 마개 lock 대기\n")
            return

        # ---------------- B-4: WAIT_BLUE ----------------
        if self.run_state == "WAIT_BLUE":
            self.wait_steps += 1
            mean = self.lock_acquirer.update()
            if mean is not None:
                self.locked_cap_center = mean
                self.sequence = build_cap_approach_sequence(self.locked_cap_center)
                self.run_state = "MOVE_TO_CAP"
                print(f"\n[B-4] blue lock 완료. fuel_cap center={np.round(mean, 4)}\n")
                return
            if self.wait_steps >= WAIT_LOCK_TIMEOUT_STEPS:
                print("\n[B-4] WAIT_BLUE timeout -> 재시도\n")
                self.wait_steps = 0
                self.lock_acquirer.reset()
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print(f"[B][WAIT_BLUE] locked={self.ros_bridge.target_locked} "
                      f"pose_count={self.ros_bridge.pose_count} lock_count={self.ros_bridge.lock_count} "
                      f"samples={len(self.lock_acquirer.samples)}/{CONTROLLER_REQUIRED_LOCK_SAMPLES}")
            return

        # ---------------- B-5: MOVE_TO_CAP ----------------
        if self.run_state == "MOVE_TO_CAP":
            self._hold_gripper(closed=False)
            done = self._drive_sequence(ee_pos)
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print("[B][MOVE_TO_CAP] " + self.sequence.debug_string(ee_pos))
            if done:
                self.sub_phase = "close_grip"
                self.gripper_hold_count = 0
                self.run_state = "GRIP_UNSCREW"
                print("\n[B-5] 마개 grasp 위치 도착 -> 그리퍼 닫기 시작\n")
            return

        # ---------------- B-6: GRIP_UNSCREW (close_grip -> rotate -> extract) ----------------
        if self.run_state == "GRIP_UNSCREW":
            if self.sub_phase == "close_grip":
                self._hold_gripper(closed=True)
                self.gripper_hold_count += 1
                if self.gripper_hold_count >= GRIPPER_ACTION_HOLD_STEPS:
                    self.joint6_start_rad = float(self.robot.get_joint_positions()[self.joint6_index])
                    self.joint6_progress_deg = 0.0
                    self.sub_phase = "rotate"
                    print("\n[B-6] 그리퍼 닫힘 -> joint_6 unscrew 회전 시작\n")
                return

            if self.sub_phase == "rotate":
                self._hold_gripper(closed=True)
                remaining = CAP_JOINT6_UNSCREW_DEG - self.joint6_progress_deg
                step_deg = np.clip(remaining, -CAP_JOINT6_DEG_PER_STEP, CAP_JOINT6_DEG_PER_STEP)
                self.joint6_progress_deg += step_deg
                set_single_joint(self.robot, self.joint6_index, self.joint6_start_rad + np.deg2rad(self.joint6_progress_deg))
                if step_count % PRINT_EVERY_N_STEPS == 0:
                    print(f"[B][GRIP_UNSCREW.rotate] progress={self.joint6_progress_deg:.1f}/{CAP_JOINT6_UNSCREW_DEG}")
                if abs(self.joint6_progress_deg - CAP_JOINT6_UNSCREW_DEG) < 1e-3:
                    self.sequence = build_cap_extract_sequence(self.locked_cap_center)
                    self.sub_phase = "extract"
                    print("\n[B-6] unscrew 360도 완료 -> 마개 빼는 중\n")
                return

            if self.sub_phase == "extract":
                self._hold_gripper(closed=True)
                done = self._drive_sequence(ee_pos)
                if step_count % PRINT_EVERY_N_STEPS == 0:
                    print("[B][GRIP_UNSCREW.extract] " + self.sequence.debug_string(ee_pos))
                if done:
                    target_joints = build_initial_joint_positions(self.robot, self.robot.get_joint_positions())
                    self._home_target_joints = target_joints
                    self._home_hold_count = 0
                    self.run_state = "RETURN_HOME_WITH_CAP"
                    print("\n[B-6] 마개 추출 완료 -> RETURN_HOME_WITH_CAP\n")
                return

        # ---------------- B-7: RETURN_HOME_WITH_CAP ----------------
        if self.run_state == "RETURN_HOME_WITH_CAP":
            self._hold_gripper(closed=True)
            reached = step_home_return(self.robot, self._home_target_joints)
            self._home_hold_count = self._home_hold_count + 1 if reached else 0
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print(f"[B][RETURN_HOME_WITH_CAP] hold={self._home_hold_count}/{HOME_HOLD_STEPS}")
            if self._home_hold_count >= HOME_HOLD_STEPS:
                self.ros_bridge.publish_robot_b_done(True)
                self.run_state = "WAIT_ROBOT_A"
                print("\n[B-7] 초기 위치 복귀(마개 보유) 완료 -> robot_b/done = True 발행\n")
            return

        # ---------------- WAIT_ROBOT_A (A-8 동안 대기) ----------------
        if self.run_state == "WAIT_ROBOT_A":
            self._hold_gripper(closed=True)
            if self._robot_a_done_received():
                self.sequence = build_cap_restore_sequence(self.task.hole_world_position)
                self.sub_phase = "insert"
                self.run_state = "RESTORE_CAP"
                print("\n[B-9] robot_a/done 수신 -> 마개 복원 시작\n")
            elif step_count % PRINT_EVERY_N_STEPS == 0:
                print("[B][WAIT_ROBOT_A] robot_a/done 대기 중")
            return

        # ---------------- B-9: RESTORE_CAP (insert -> screw -> open_grip) ----------------
        if self.run_state == "RESTORE_CAP":
            if self.sub_phase == "insert":
                self._hold_gripper(closed=True)
                done = self._drive_sequence(ee_pos)
                if step_count % PRINT_EVERY_N_STEPS == 0:
                    print("[B][RESTORE_CAP.insert] " + self.sequence.debug_string(ee_pos))
                if done:
                    self.joint6_start_rad = float(self.robot.get_joint_positions()[self.joint6_index])
                    self.joint6_progress_deg = 0.0
                    self.sub_phase = "screw"
                    print("\n[B-9] 마개 삽입 위치 도착 -> joint_6 screw 회전 시작\n")
                return

            if self.sub_phase == "screw":
                self._hold_gripper(closed=True)
                remaining = CAP_JOINT6_SCREW_DEG - self.joint6_progress_deg
                step_deg = np.clip(remaining, -CAP_JOINT6_DEG_PER_STEP, CAP_JOINT6_DEG_PER_STEP)
                self.joint6_progress_deg += step_deg
                set_single_joint(self.robot, self.joint6_index, self.joint6_start_rad + np.deg2rad(self.joint6_progress_deg))
                if step_count % PRINT_EVERY_N_STEPS == 0:
                    print(f"[B][RESTORE_CAP.screw] progress={self.joint6_progress_deg:.1f}/{CAP_JOINT6_SCREW_DEG}")
                if abs(self.joint6_progress_deg - CAP_JOINT6_SCREW_DEG) < 1e-3:
                    self.gripper_hold_count = 0
                    self.sub_phase = "open_grip"
                    print("\n[B-9] screw 360도 완료 -> 그리퍼 열기\n")
                return

            if self.sub_phase == "open_grip":
                self._hold_gripper(closed=False)
                self.gripper_hold_count += 1
                if self.gripper_hold_count >= GRIPPER_ACTION_HOLD_STEPS:
                    self.sequence = build_cover_sequence(
                        "close", self.locked_door_center, self.task.door_pivot_world, self.task.door_axis_world,
                    )
                    self.sub_phase = None
                    self.run_state = "CLOSE_COVER"
                    print("\n[B-9] 마개 복원 완료 -> CLOSE_COVER\n")
            return

        # ---------------- B-10: CLOSE_COVER ----------------
        if self.run_state == "CLOSE_COVER":
            done = self._drive_sequence(ee_pos, extra_condition_ok=self._door_angle_ok())
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print(f"[B][CLOSE_COVER] door_angle={self.task.current_door_angle_deg():.1f} "
                      + self.sequence.debug_string(ee_pos))
            if done:
                target_joints = build_initial_joint_positions(self.robot, self.robot.get_joint_positions())
                self._home_target_joints = target_joints
                self._home_hold_count = 0
                self.run_state = "FINAL_HOME"
                print("\n[B-10] 커버 닫힘(30도) 완료 -> FINAL_HOME\n")
            return

        # ---------------- B-11: FINAL_HOME ----------------
        if self.run_state == "FINAL_HOME":
            reached = step_home_return(self.robot, self._home_target_joints)
            self._home_hold_count = self._home_hold_count + 1 if reached else 0
            if step_count % PRINT_EVERY_N_STEPS == 0:
                print(f"[B][FINAL_HOME] hold={self._home_hold_count}/{HOME_HOLD_STEPS}")
            if self._home_hold_count >= HOME_HOLD_STEPS:
                self.task_done = True
                print("\n[B-11] 초기 위치 복귀 완료. 전체 시퀀스 종료.\n")
            return


# ╔══════════════════════════════════════════════════════════════╗
# ║  F. 메인                                                       ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    my_world = World(stage_units_in_meters=1.0)
    task = MultiRobotOilingTask(name="multi_robot_oiling_task")
    my_world.add_task(task)
    my_world.reset()

    robot_a = my_world.scene.get_object("m0609_A")
    robot_b = my_world.scene.get_object("m0609_B")

    initialize_robot(robot_a, my_world, ROBOT_A_BASE_WORLD, ROBOT_A_BASE_ORIENTATION)
    initialize_robot(robot_b, my_world, ROBOT_B_BASE_WORLD, ROBOT_B_BASE_ORIENTATION)

    for _ in range(30):
        my_world.step(render=True)

    print("\n" + "=" * 60)
    print("[C-1] 초기 상태")
    print("=" * 60)
    print(f"  m0609_A base = {ROBOT_A_BASE_WORLD}, joints(deg)={INITIAL_ARM_JOINT_DEG}")
    print(f"  m0609_B base = {ROBOT_B_BASE_WORLD}, joints(deg)={INITIAL_ARM_JOINT_DEG}")

    controller_a = RMPFlowController(
        name="m0609_A_rmpflow_controller",
        robot_articulation=robot_a,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )
    controller_b = RMPFlowController(
        name="m0609_B_rmpflow_controller",
        robot_articulation=robot_b,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name=EE_LINK_NAME,
    )
    print("  [OK] RMPFlowController A/B 생성 완료")

    if not rclpy.ok():
        rclpy.init(args=None)
    ros_bridge = MultiRobotRosBridge()
    camera_prim_path = find_camera_prim_path()
    print(f"  [ROS] camera_prim_path = {camera_prim_path}")

    runner_a = RobotARunner(robot_a, controller_a, ros_bridge, camera_prim_path, task)
    runner_b = RobotBRunner(robot_b, controller_b, ros_bridge, camera_prim_path, task)

    was_playing = False
    step_count = 0

    while simulation_app.is_running():
        my_world.step(render=True)
        time.sleep(0.01)
        is_playing = my_world.is_playing()
        if rclpy.ok():
            rclpy.spin_once(ros_bridge, timeout_sec=0.0)

        if is_playing and not was_playing:
            my_world.reset()
            initialize_robot(robot_a, my_world, ROBOT_A_BASE_WORLD, ROBOT_A_BASE_ORIENTATION)
            initialize_robot(robot_b, my_world, ROBOT_B_BASE_WORLD, ROBOT_B_BASE_ORIENTATION)
            controller_a.reset()
            controller_b.reset()
            task.post_reset()
            runner_a.on_play_reset()
            runner_b.on_play_reset()
            step_count = 0
            print("\n[RESET] multi-robot oiling sequence 준비 완료\n")

        if is_playing:
            step_count += 1
            runner_a.tick(step_count)
            runner_b.tick(step_count)

            if runner_a.task_done and runner_b.task_done:
                print("\n[완료] A/B 전체 시퀀스 종료 - 시뮬레이션 일시정지\n")
                my_world.pause()

        was_playing = is_playing

    if rclpy.ok():
        ros_bridge.destroy_node()
        rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    main()
