from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
# 창 생성
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")
simulation_app.update()
# 브릿지 켜기

from dataclasses import dataclass
# 데이터 묶음을 간편하게 만드는 도구
from pathlib import Path
import sys
import time

import numpy as np
import omni.usd
# 아이작심에 열려있는 씬에 접근할때 사용
from pxr import Usd, UsdGeom, UsdPhysics, Gf
# Usd : 부품 순회
# UsdGeom : 공간및 도형 생성
# UsdPhysics : 관절 드라이브
# Gf : 백터 생성
# Prim >> USD에서 모든 객체의 기본 단위

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid
# 물체 생성 
from isaacsim.core.api.tasks import BaseTask
from isaacsim.core.utils.rotations import euler_angles_to_quat 
# 오일러각을 쿼터니언으로 변환
from isaacsim.robot.manipulators.grippers import ParallelGripper 
# 그리퍼 제어
from isaacsim.robot.manipulators.manipulators import SingleManipulator

_THIS_DIR = Path(__file__).resolve().parent
# 현재 파일의 폴더 위치

# rmpflow 인프라 폴더 경로 등록
RMPFLOW_DIR = str(_THIS_DIR / "rmpflow")


if RMPFLOW_DIR not in sys.path:
    sys.path.insert(0, RMPFLOW_DIR)

# Pick & Place 예제와 같은 RMPFlow 인프라 사용.
# 단, PickPlaceController의 pick/grip/place 상태머신은 쓰지 않고,
# RMPFlowController에 waypoint를 직접 주는 구조로 자동 주유/삽입 시퀀스를 구성한다.
from m0609_rmpflow_controller import RMPFlowController


# ╔══════════════════════════════════════════════════════════════╗
# ║  A. 기존 Pick & Place 코드 기반 환경 파라미터                  ║
# ╚══════════════════════════════════════════════════════════════╝
USD_PATH        = str(_THIS_DIR / "Collected_m0609_camera/m0609_camera.usd")
# USD파일 경로 : Collected로 저장이 되어 있지 않다면 오류남
ROBOT_PRIM_PATH = "/World/m0609"
EE_LINK_NAME    = "link_6"       # nozzle_tip 추가 전까지는 link_6 기준 제어 (Isaac Sim의 자체 Gripper 기준은 link_6 유지)
GRIPPER_JOINTS  = ["finger_joint", "right_inner_knuckle_joint"]

DRIVE_STIFFNESS = 1e8 # 강도
DRIVE_DAMPING   = 1e4 # 완충제  부드럽게 멈춤
DRIVE_MAX_FORCE = 1e8 # 최대 하중

GRIPPER_OPEN    = [0.0, 0.0] # 그리퍼 열린 상태
GRIPPER_CLOSE   = [0.5, 0.5] # 그리퍼 닫힌 상태
GRIPPER_DELTA   = [-0.5, -0.5] # 그리퍼 상태 변환용

# RMPFlow 설정 파일 경로
M0609_URDF_PATH           = str(_THIS_DIR / "doosan-robot2/urdf/m0609_isaac_sim.urdf") 
M0609_DESCRIPTION_PATH    = str(_THIS_DIR / "rmpflow/m0609_description.yaml")
M0609_RMPFLOW_CONFIG_PATH = str(_THIS_DIR / "rmpflow/m0609_rmpflow_common.yaml")


# ╔══════════════════════════════════════════════════════════════╗
# ║  B. 벽 부착형 자동 주유 테스트 파라미터                         ║
# ╚══════════════════════════════════════════════════════════════╝
# 목표: 로봇 베이스를 벽면에 부착된 형태로 두고, y=-0.95 위치의 실린더 주유구에 삽입한다.
ROBOT_BASE_WORLD = np.array([0.0, 0.0, 1.0], dtype=float) # 로봇 베이스 좌표 >> z축으로 1미터
ROBOT_BASE_EULER_DEG = np.array([90.0, 0.0, 0.0], dtype=float) # x축으로 90도 (DEG ~도의미)
ROBOT_BASE_ORIENTATION = euler_angles_to_quat(np.deg2rad(ROBOT_BASE_EULER_DEG)) # 쿼터니언으로 변환

# 쿼터니언으로 변환과정
# 도 >> 라디안 >> 쿼터니언 

# 사용자가 지정한 초기 관절각
# Isaac/URDF 관절 명령은 radian이므로 내부에서 deg->rad 변환한다.
INITIAL_ARM_JOINT_DEG = {
    "joint_1": 10.0,
    "joint_2": -66.0,
    "joint_3": 150.0,
    "joint_4": 3.5,
    "joint_5": -75.0,
    # 카메라/그리퍼가 뒤집혀 보이는 문제를 줄이기 위해 wrist roll을 180도 뒤집는다.
    # 기존 175도 + 180도 = 355도이며, 같은 의미로 -5도로 입력한다.
    "joint_6": -5.0,
}
INITIAL_GRIPPER_JOINTS = {
    "finger_joint": 0.0,
    "right_inner_knuckle_joint": 0.0,
}

# 목표 실린더 주유구: 0.1 x 0.1 x 0.1 크기
# UsdGeom.Cylinder 기준 radius=0.05, height=0.10이다.
FUEL_PORT_CENTER = np.array([0.0, -0.95, 1.0], dtype=float) 
# 주유구 입구
FUEL_PORT_EULER_DEG = np.array([90.0, 0.0, 0.0], dtype=float)  
# 각도 베이스와 같은 90도 방향
FUEL_PORT_DIAMETER = 0.10 # 지름 
FUEL_PORT_RADIUS = FUEL_PORT_DIAMETER / 2.0 # 반지름
FUEL_PORT_DEPTH = 0.10 # 깊이

# 벽면 실린더는 local Z축 cylinder를 X축 90도 회전시켜, 축이 대략 -Y 방향을 향하게 둔다.
# 로봇이 y=0 쪽에 있고 목표가 y=-1.05이므로,
# 주유구 바깥쪽 normal은 +Y, 삽입 방향은 -Y로 고정한다.
PORT_OUTWARD_NORMAL = np.array([0.0, 1.0, 0.0], dtype=float)
# 주유구가 로봇을 바라보는 방향
INSERTION_DIRECTION = np.array([0.0, -1.0, 0.0], dtype=float)
# 노즐이 주유구 안으로 들어가는 방향

# 접근 거리. 모두 실린더 중심 기준 거리이다.
FAR_DISTANCE     = 0.26
MID_DISTANCE     = 0.17
NEAR_DISTANCE    = 0.085
# 실린더 중심에서 -Y 방향으로 4.5cm 들어간 위치. depth/2=5cm보다 살짝 작게 둔다.
INSERT_DISTANCE  = FUEL_PORT_DEPTH / 2.0 - 0.005


# waypoint 판정 기준
POSITION_TOLERANCE = 0.035
# 목표에 도달했다고 판정하는 허용 오차
MAX_STEPS_PER_STAGE = 1800
# 한 단계에서 최대 몇스텝까지 기다릴지 1800 /60 >> 30초
PRINT_EVERY_N_STEPS = 20
# 몇스탭마다 터미널에 디버그 출력할지


# 속도 제한: RMPFlow 목표점을 바로 주지 않고, 가상의 command target을 조금씩 이동시킨다.
PHYSICS_DT = 1.0 / 60.0
# 1초에 60번 업데이트 
DEFAULT_TARGET_SPEED = 0.060
# 일반 이동 속도 : 0.06 * 60 = 3.6m/s
NEAR_TARGET_SPEED    = 0.040
# 근처 이동 : 2.4m/s
INSERT_TARGET_SPEED  = 0.020
# 삽입 : 1.2m/s
RETREAT_TARGET_SPEED = 0.050
# 후퇴 : 3m/s

# 복귀는 사용자가 지정한 초기 자세로 돌아간다.
HOME_JOINT_SPEED_ALPHA = 0.012 # 보간법 처음에는 빨리 점점 천천히
HOME_JOINT_TOLERANCE = 0.035 # 오차값
HOME_HOLD_STEPS = 80 # 80/60 >>1.3초 안정화 대기후 종료 

# link_6 위치만 제어하면 손목/그리퍼가 기울어진 채로 접근하여 충돌할 수 있다.
# 하지만 euler 고정값을 중간 stage부터 갑자기 강제하면 손목이 과하게 정렬될 수 있다.
# 따라서 Play/reset 직후의 EE orientation을 "삽입 축 정렬 자세"로 잠그고,
# 모든 접근/삽입/후퇴 구간에서 같은 orientation을 유지한다.
# link_5를 직접 EE로 바꾸지는 않는다. 대신 link_5->link_6 축이 삽입축과 얼마나 어긋나는지 로그로 감시한다.
USE_TARGET_ORIENTATION = True
TARGET_ORIENTATION = None  # main loop에서 초기 EE orientation으로 설정


# ============================================================
# 유틸
# ============================================================
def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n
# 뱡향만 알고 싶어서 길이를 1로 만들어서 계산 

def find_prim_path_by_name(root_path: str, name: str):
    stage = omni.usd.get_context().get_stage()
    # 도면을 가져옴 
    root_prim = stage.GetPrimAtPath(root_path)
    # 도면에서 특정 경로의 부품을 찾음
    # root_path = "/World/m0609" 이면 로봇 전체를 가져옴 
    if not root_prim.IsValid():
        return None
    # 유효하지 않다면 None
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return str(prim.GetPath())
    # 유효하다면 우리가 필요한 prim의 주소를 가져옴    
    return None


def get_prim_world_position(prim_path: str) -> np.ndarray | None:
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.XformCache()
    # 변환행렬을 저장하는 도구 
    mat = cache.GetLocalToWorldTransform(prim)
    # 월드 좌표의 행렬 가져옴
    t = mat.ExtractTranslation()
    # mat = [회전|위치]  →  t = [x, y, z]
    # 위치만 필요하니까 ExtractTranslation()으로 꺼냄
    return np.array([float(t[0]), float(t[1]), float(t[2])], dtype=float)

# 5번 조인트와 6번 조인트의 방향이 얼마나 틀어져 있는지 알려주는 함수
# 직선으로 움직이기 위해서 두 조인트를 동일하게 움직이게 하기 위해서 
# 두 백터사이의 각도를 -1, 1사이로 강제 제한 
def angle_deg_between(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-9) -> float | None:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return None
    c = float(np.clip(np.dot(v1 / n1, v2 / n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


# 관절이름으로 인덱스를 찾는 함수
def find_dof_index(robot, dof_name: str):
    if hasattr(robot, "dof_names") and dof_name in robot.dof_names:
        # # robot 객체에 dof_names 속성이 있는지 확인
        # 없으면 에러 방지
        return robot.dof_names.index(dof_name)
    return None

# 초기 관절값 배열을 만드는 함수 
def build_initial_joint_positions(robot, base_positions=None) -> np.ndarray:
    """현재 robot.num_dof에 맞춰 초기 관절 벡터를 만든다."""
    if base_positions is None:
        q = np.zeros(robot.num_dof, dtype=float)
    # 빈 배열 생성 
    else:
        q = np.array(base_positions, dtype=float).copy()
        if len(q) != robot.num_dof:
            q = np.zeros(robot.num_dof, dtype=float)

    # 1차: dof_names로 정확히 매핑
    missing_arm = []
    for joint_name, deg in INITIAL_ARM_JOINT_DEG.items():
        idx = find_dof_index(robot, joint_name)
        # 인덱스 찾기
        if idx is None:
            missing_arm.append(joint_name)
        # 못찾으면 리스트에 넣어주기
        else:
            q[idx] = np.deg2rad(deg)
        # 라디안 값으로 바꿔서 조인트값 지정

    # 방어: dof_names가 다르게 들어오는 경우, 앞 6개를 arm joint로 가정해 fallback
    if missing_arm and robot.num_dof >= 6:
        fallback_values = [
            INITIAL_ARM_JOINT_DEG["joint_1"],
            INITIAL_ARM_JOINT_DEG["joint_2"],
            INITIAL_ARM_JOINT_DEG["joint_3"],
            INITIAL_ARM_JOINT_DEG["joint_4"],
            INITIAL_ARM_JOINT_DEG["joint_5"],
            INITIAL_ARM_JOINT_DEG["joint_6"],
        ]
        for i, deg in enumerate(fallback_values):
            q[i] = np.deg2rad(deg)

    # gripper는 0으로 열린 상태 유지
    for joint_name, value in INITIAL_GRIPPER_JOINTS.items():
        idx = find_dof_index(robot, joint_name)
        if idx is not None:
            q[idx] = value

    return q


def apply_robot_start_state(robot):
    """벽 부착형 root pose와 사용자가 지정한 초기 관절각을 적용문을 적용한다."""
    # 로봇 포지션 변경
    robot.set_world_pose(
        position=ROBOT_BASE_WORLD,
        orientation=ROBOT_BASE_ORIENTATION,
    )
    # 조인트값 가져오기 
    current = robot.get_joint_positions()
    # 각 관절값 지정 
    q0 = build_initial_joint_positions(robot, current)
    # 관절값 적용 
    robot.set_joint_positions(q0)
    return q0

# 로봇 초기 세팅
def initialize_robot(robot, world):
    robot.initialize()
    robot.gripper.initialize(
        physics_sim_view=world.physics_sim_view,
        articulation_apply_action_func=robot.apply_action,
        get_joint_positions_func=robot.get_joint_positions,
        set_joint_positions_func=robot.set_joint_positions,
        dof_names=robot.dof_names,
    )
    q0 = apply_robot_start_state(robot)
    robot.gripper.set_joint_positions(np.array(GRIPPER_OPEN, dtype=float))
    return q0

# 실린더 생성 코드 
def create_usd_visual_cylinder(prim_path: str, position: np.ndarray, radius: float, height: float,
                               euler_deg: np.ndarray, color: np.ndarray):
    """VisualCylinder import 호환성 이슈를 피하기 위해 pxr UsdGeom.Cylinder로 직접 생성한다."""
    stage = omni.usd.get_context().get_stage()
    cyl = UsdGeom.Cylinder.Define(stage, prim_path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr(UsdGeom.Tokens.z)
    # z축 방향으로 세운다
    cyl.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    # 투명도
    cyl.CreateDisplayOpacityAttr([0.85])

    xform = UsdGeom.Xformable(cyl.GetPrim())
    # xfrom >> 실린더 prim을 Xformable로 감싸는 것
    # Xformable = 위치/회전/크기를 설정할 수 있는 객체
    xform.ClearXformOpOrder()
    # 기존 설정 초기화 
    xform.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(float(euler_deg[0]), float(euler_deg[1]), float(euler_deg[2])))
    return cyl


@dataclass
class FuelStage:
    name: str
    target_position: np.ndarray | None
    hold_steps: int = 0
    tolerance: float = POSITION_TOLERANCE
    max_steps: int = MAX_STEPS_PER_STAGE
    speed: float = DEFAULT_TARGET_SPEED
    use_orientation: bool = False


class FuelPortSequence:
    """벽면 실린더 주유구에 대한 자동 주유 waypoint state machine."""

    def __init__(self):
        self.port_outward_normal = normalize(PORT_OUTWARD_NORMAL)
        self.insertion_direction = normalize(INSERTION_DIRECTION)

        # tip waypoint: 실제로 실린더에 들어간다고 가정하는 가상 nozzle_tip 목표.
        tip_approach_far  = FUEL_PORT_CENTER + self.port_outward_normal * FAR_DISTANCE
        tip_approach_mid  = FUEL_PORT_CENTER + self.port_outward_normal * MID_DISTANCE
        tip_approach_near = FUEL_PORT_CENTER + self.port_outward_normal * NEAR_DISTANCE
        tip_insert_target = FUEL_PORT_CENTER + self.insertion_direction * INSERT_DISTANCE

        # control waypoint: 현재 RMPFlow EE frame은 link_6이므로, link_6 목표는 tip 목표보다 바깥쪽으로 물러나야 한다.
        # nozzle_tip = link_6 - port_outward_normal * VIRTUAL_NOZZLE_LENGTH 라고 가정한다.
        # [수정 포인트] RMPflow가 직접 제어하므로 VIRTUAL_NOZZLE_LENGTH가 0.0이 되어 control_offset은 이제 [0,0,0]입니다.
        control_offset = self.port_outward_normal * VIRTUAL_NOZZLE_LENGTH + np.array([0.0, 0.0, VIRTUAL_NOZZLE_Z_OFFSET])

        # 가상 위치 적용 (수정 후에는 tip_target과 완벽하게 동일한 값이 됩니다)
        approach_far  = tip_approach_far + control_offset
        approach_mid  = tip_approach_mid + control_offset
        approach_near = tip_approach_near + control_offset
        insert_target = tip_insert_target + control_offset


        # 
        self.tip_targets = {
            "01_axis_far_start": tip_approach_far,
            "02_axis_mid": tip_approach_mid,
            "03_axis_near_stop": tip_approach_near,
            "04_insert_into_cylinder": tip_insert_target,
            "05_retreat_near": tip_approach_near,
            "06_retreat_mid": tip_approach_mid,
            "07_retreat_far": tip_approach_far,
        }

        self.stages = [
            # 벽 부착형 시나리오: 위쪽 점을 찍지 않고 처음부터 주유구 축과 같은 선상으로 직선 접근한다.
            # orientation은 main loop에서 reset 직후 EE 자세로 잠근 뒤 전 구간 유지한다.
            FuelStage("01_axis_far_start", approach_far, tolerance=0.030, speed=DEFAULT_TARGET_SPEED, use_orientation=True),
            #단계 이름, 목표위치, 접근 위치, 속도, 위치 고정
            FuelStage("02_axis_mid", approach_mid, tolerance=0.028, speed=DEFAULT_TARGET_SPEED, use_orientation=True),
            FuelStage("03_axis_near_stop", approach_near, hold_steps=80, tolerance=0.022, speed=NEAR_TARGET_SPEED, use_orientation=True),
            FuelStage("04_insert_into_cylinder", insert_target, hold_steps=180, tolerance=0.018, speed=INSERT_TARGET_SPEED, use_orientation=True),

            # 후퇴도 같은 orientation을 유지한 채 역순으로 빠져나온다.
            FuelStage("05_retreat_near", approach_near, speed=RETREAT_TARGET_SPEED, use_orientation=True),
            FuelStage("06_retreat_mid", approach_mid, speed=RETREAT_TARGET_SPEED, use_orientation=True),
            FuelStage("07_retreat_far", approach_far, hold_steps=30, tolerance=0.035, speed=RETREAT_TARGET_SPEED, use_orientation=True),

            # RMPFlow 위치 제어가 아니라 joint 직접 보간으로 초기 자세 복귀
            FuelStage("08_return_home", None, hold_steps=HOME_HOLD_STEPS, use_orientation=False),
        ]
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

    def update(self, ee_position: np.ndarray, ee_orientation: np.ndarray = None) -> bool:
        """현재 stage 완료 여부를 판단하고, 필요하면 다음 stage로 전환한다."""
        if self.done:
            return True

        stage = self.current
        self.stage_step += 1

        reached = False
        # 도착 여부
        if stage.target_position is None:
            # return_home은 main loop에서 별도 처리하므로 여기서 자동 완료시키지 않는다.
            reached = False
        else:
            err = np.linalg.norm(stage.target_position - ee_position)
            reached = err < stage.tolerance

        timed_out = self.stage_step >= stage.max_steps

        if reached:
            self.hold_count += 1
        else:
            self.hold_count = 0
            
        STAGE_LOG = {
            "01_axis_far_start":       "접근 시작",
            "02_axis_mid":             "중간 지점 도착",
            "03_axis_near_stop":       "근처 도착 대기",
            "04_insert_into_cylinder": "삽입 완료",
            "05_retreat_near":         "후퇴 시작",
            "06_retreat_mid":          "중간 후퇴",
            "07_retreat_far":          "먼 후퇴 완료",
            "08_return_home":          "홈 복귀 완료",
        }

        if (reached and self.hold_count >= stage.hold_steps) or timed_out:
            if timed_out and not reached:
                print(f"\n[⚠️ 타임아웃] {STAGE_LOG.get(stage.name, stage.name)}")
                print(f"  위치 = {np.round(ee_position, 3)}")
                print(f"  목표까지 = {np.linalg.norm(stage.target_position - ee_position):.3f}m 남음\n")
            else:
                print(f"\n[✅ {STAGE_LOG.get(stage.name, stage.name)}]")
                print(f"  위치  = {np.round(ee_position, 3)}")
                if ee_orientation is not None:
                    print(f"  각도  = {np.round(ee_orientation, 3)}\n")

            self.index += 1
            self.stage_step = 0
            self.hold_count = 0
            self.command_target = None
            if self.index >= len(self.stages):
                self.done = True
                return True
        return self.done

    def get_command_target(self, ee_position: np.ndarray) -> np.ndarray | None:
        """실제 stage 목표까지 한 번에 보내지 않고, 속도 제한된 중간 목표를 반환한다."""
        if self.done:
            return None
        stage = self.current
        if stage.target_position is None:
            return None
        if self.command_target is None:
            self.command_target = np.array(ee_position, dtype=float)

        delta = stage.target_position - self.command_target
        # 목표 까지의 백터 
        dist = np.linalg.norm(delta)
        # 목표 거리 
        max_step = max(stage.speed * PHYSICS_DT, 1e-5)
        # 이번 스탭에 최대로 이동할 수 있는 거리 
        if dist <= max_step:
            self.command_target = np.array(stage.target_position, dtype=float)
        else:
            self.command_target = self.command_target + delta / dist * max_step
        return self.command_target

    def debug_string(self, ee_position: np.ndarray) -> str:
        if self.done:
            return "[DONE] fuel sequence complete"
        stage = self.current
        if stage.target_position is None:
            return f"[stage={self.index}:{stage.name}] hold={self.hold_count}/{stage.hold_steps}"
        err = np.linalg.norm(stage.target_position - ee_position)
        # 남은거리 
        cmd = self.command_target if self.command_target is not None else np.array([np.nan, np.nan, np.nan])
        tip = self.tip_targets.get(stage.name, None)
        tip_str = "None" if tip is None else str(np.round(tip, 3))
        return (
            f"[stage={self.index}:{stage.name}] "
            f"link6_target={np.round(stage.target_position, 3)} "
            f"tip_target={tip_str} "
            f"cmd={np.round(cmd, 3)} "
            f"ee={np.round(ee_position, 3)} "
            f"err={err:.4f} "
            f"speed={stage.speed:.3f} "
            f"ori={'ON' if stage.use_orientation else 'OFF'} "
            f"hold={self.hold_count}/{stage.hold_steps}"
        )


# ============================================================
# Task — Pick & Place 구조 유지, 작업물만 벽면 실린더 주유구로 변경
# ============================================================
class M0609FuelPortWallCylinderTask(BaseTask):

    def __init__(self, name):
        super().__init__(name=name, offset=None)
        self.sequence = FuelPortSequence()

    def set_up_scene(self, scene):
        super().set_up_scene(scene)
        self._load_usd()
        self._discover_links()
        self._setup_physics()
        self._register_robot(scene)
        self._create_fuel_port_scene(scene)
        print("\n  [완료] 벽 부착형 자동 주유 테스트 씬 구성 성공!\n")
    
    # usd 로드
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

    # 링크 발견 >> 링크 주소 저장
    def _discover_links(self):
        print("\n" + "=" * 60)
        print("[2.DISCOVER] 링크 경로 탐색")
        print("=" * 60)
        self._ee_path = find_prim_path_by_name(ROBOT_PRIM_PATH, EE_LINK_NAME)
        self._link5_path = find_prim_path_by_name(ROBOT_PRIM_PATH, "link_5")
        self._link6_path = find_prim_path_by_name(ROBOT_PRIM_PATH, "link_6")
        if self._ee_path is None:
            raise RuntimeError(f"'{EE_LINK_NAME}' not found under {ROBOT_PRIM_PATH}")
        print(f"  EE ({EE_LINK_NAME}) = {self._ee_path}")
        print(f"  link_5 monitor = {self._link5_path}")
        print(f"  link_6 monitor = {self._link6_path}")
        for jn in GRIPPER_JOINTS:
            print(f"  {jn:<35} = {find_prim_path_by_name(ROBOT_PRIM_PATH, jn)}")

    # 로봇에 강도, 완충, 하중 설정 
    def _setup_physics(self):
        print("\n" + "=" * 60)
        print("[3.PHYSICS] 로봇 drive 설정")
        print("=" * 60)
        stage = omni.usd.get_context().get_stage()

        drive_count = 0
        root = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        if not root.IsValid():
            raise RuntimeError(f"Robot prim not found: {ROBOT_PRIM_PATH}")

        for prim in Usd.PrimRange(root):
            for dt in ["angular", "linear"]:
                drive = UsdPhysics.DriveAPI.Get(prim, dt)
                if drive:
                    drive.GetStiffnessAttr().Set(DRIVE_STIFFNESS)
                    drive.GetDampingAttr().Set(DRIVE_DAMPING)
                    drive.GetMaxForceAttr().Set(DRIVE_MAX_FORCE)
                    drive_count += 1
        print(f"  [OK] drive updated: {drive_count}")

    # 로봇 생성
    def _register_robot(self, scene):
        print("\n" + "=" * 60)
        print("[4.REGISTER] SingleManipulator 등록")
        print("=" * 60)
        gripper = ParallelGripper(
            end_effector_prim_path=self._ee_path,
            joint_prim_names=GRIPPER_JOINTS,
            joint_opened_positions=np.array(GRIPPER_OPEN),
            joint_closed_positions=np.array(GRIPPER_CLOSE),
            action_deltas=np.array(GRIPPER_DELTA),
        )
        self._robot = scene.add(
            SingleManipulator(
                prim_path=ROBOT_PRIM_PATH,
                name="m0609_robot",
                end_effector_prim_path=self._ee_path,
                gripper=gripper,
            )
        )
        print(f"  [OK] SingleManipulator: {ROBOT_PRIM_PATH}")

    # 씬구상
    def _create_fuel_port_scene(self, scene):
        print("\n" + "=" * 60)
        print("[5.SCENE] 벽면 실린더 주유구 + waypoint 생성")
        print("=" * 60)

        # 실린더 주유구 생성. radius 0.05, height 0.10 -> 0.1 x 0.1 x 0.1 크기.
        create_usd_visual_cylinder(
            prim_path="/World/fuel_port_cylinder",
            position=FUEL_PORT_CENTER,
            radius=FUEL_PORT_RADIUS,
            height=FUEL_PORT_DEPTH,
            euler_deg=FUEL_PORT_EULER_DEG,
            color=np.array([0.0, 1.0, 0.0]),
        )
        print(f"  [OK] 주유구 실린더 중심 @ {FUEL_PORT_CENTER}")
        print(f"  [OK] 실린더 사이즈 = 지름 {FUEL_PORT_DIAMETER}, 깊이 {FUEL_PORT_DEPTH}")
        print(f"  [INFO] 주유구 각도 = {FUEL_PORT_EULER_DEG}")
        print(f"  [INFO] 주유구 바깥 방향 = {np.round(self.sequence.port_outward_normal, 4)}")
        print(f"  [INFO] 주유구 삽입 방향 = {np.round(self.sequence.insertion_direction, 4)}")
        print(f"  [INFO] link_6에서 노즐까지의 거리 = {VIRTUAL_NOZZLE_LENGTH} m")


        # 입구면 중심 표시: 노란색 판 생성
        mouth_center = FUEL_PORT_CENTER + self.sequence.port_outward_normal * (FUEL_PORT_DEPTH / 2.0)
        scene.add(
            VisualCuboid(
                prim_path="/World/fuel_port_mouth_center",
                name="fuel_port_mouth_center",
                position=mouth_center,
                scale=np.array([0.035, 0.006, 0.035]),
                color=np.array([1.0, 1.0, 0.0]),
            )
        )
        print(f"  [OK] 주유구 마커 생성 @ {np.round(mouth_center, 4)}")

        # waypoint marker 생성
        for i, stage in enumerate(self.sequence.stages):
            if stage.target_position is None:
                continue
            if "insert" in stage.name:
                color = np.array([1.0, 0.0, 1.0])
                scale = np.array([0.035, 0.035, 0.035])
            elif "retreat" in stage.name:
                color = np.array([0.0, 1.0, 0.3])
                scale = np.array([0.025, 0.025, 0.025])
            elif "near" in stage.name:
                color = np.array([1.0, 0.5, 0.0])
                scale = np.array([0.030, 0.030, 0.030])
            else:
                color = np.array([0.0, 0.3, 1.0])
                scale = np.array([0.025, 0.025, 0.025])

            # link_6 control target marker
            scene.add(
                VisualCuboid(
                    prim_path=f"/World/fuel_wp_{i:02d}_{stage.name}",
                    # 02d 숫자 두자리로 표시
                    # 왜? >> 그렇게 적어 두어서
                    name=f"fuel_wp_{i:02d}_{stage.name}",
                    position=stage.target_position,
                    scale=scale,
                    color=color,
                )
            )

            # 가상 nozzle_tip marker: 실제 실린더에 들어가는 점을 따로 표시한다.
            tip_target = self.sequence.tip_targets.get(stage.name)
            if tip_target is not None:
                scene.add(
                    VisualCuboid(
                        prim_path=f"/World/fuel_tip_wp_{i:02d}_{stage.name}",
                        name=f"fuel_tip_wp_{i:02d}_{stage.name}",
                        position=tip_target,
                        scale=np.array([0.015, 0.015, 0.015]),
                        color=np.array([1.0, 1.0, 1.0]),
                    )
                )

            print(
                f"  [OK] {i:02d} {stage.name:<28} "
                f"link6_target={np.round(stage.target_position, 4)} "
                f"tip_target={np.round(tip_target, 4) if tip_target is not None else None}"
            )
    
    # 로봇의 현재 상태 추적
    def get_observations(self):
        ee_pos, ee_ori = self._robot.end_effector.get_world_pose()
        link5_pos = get_prim_world_position(getattr(self, "_link5_path", ""))
        link6_pos = get_prim_world_position(getattr(self, "_link6_path", ""))
        # 월드좌표 가져오기 
        link5_to_link6_angle = None
        if link5_pos is not None and link6_pos is not None:
            link5_to_link6_axis = link6_pos - link5_pos
            # link_5에서 link_6으로 향하는 방향 벡터
            link5_to_link6_angle = angle_deg_between(link5_to_link6_axis, INSERTION_DIRECTION)
            # 삽입 방향과 몇도 정도 차이나는지
        return {
            self._robot.name: {
                "joint_positions": self._robot.get_joint_positions(),
                "ee_position": ee_pos,
                "ee_orientation": ee_ori,
                "link5_to_link6_angle_deg": link5_to_link6_angle,
            },
            "fuel_port": {
                "center": FUEL_PORT_CENTER,
                "outward_normal": self.sequence.port_outward_normal,
                "insertion_direction": self.sequence.insertion_direction,
            },
        }

    def post_reset(self):
        self._robot.gripper.set_joint_positions(
            self._robot.gripper.joint_opened_positions
        )
        self.sequence.reset()


# ╔══════════════════════════════════════════════════════════════╗
# ║  C. 메인 — RMPFlow waypoint 제어                              ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    my_world = World(stage_units_in_meters=1.0)
    # 월드 단위 1미터로 설정
    task = M0609FuelPortWallCylinderTask(name="m0609_fuel_port_wall_cylinder_task")
    # tset설정
    my_world.add_task(task)
    # 테스트 넣어주기
    my_world.reset()

    robot = my_world.scene.get_object("m0609_robot")
    # 로봇 가져오기
    q0 = initialize_robot(robot, my_world)
    # 설정한 관절값들 가져오기

    # 홈 포지션 안정화 대기
    for _ in range(30):
        my_world.step(render=True)

    print("\n" + "=" * 60)
    print("[C-1] 초기 상태")
    print("=" * 60)
    print(f"  robot_base_world     = {ROBOT_BASE_WORLD}")
    print(f"  robot_base_euler_deg = {ROBOT_BASE_EULER_DEG}")
    print(f"  dof_names            = {robot.dof_names}")
    print(f"  initial_q(rad)       = {np.round(q0, 4)}")
    print(f"  initial_q(deg arm)   = {INITIAL_ARM_JOINT_DEG}")

    print("\n" + "=" * 60)
    print("[C-2] RMPFlowController 생성")
    print("=" * 60)
    print(f"  URDF        = {M0609_URDF_PATH}")
    print(f"  description = {M0609_DESCRIPTION_PATH}")
    print(f"  rmpflow     = {M0609_RMPFLOW_CONFIG_PATH}")
    print(f"  EE frame    = {EE_LINK_NAME}")
    print(f"  orientation = {'ON' if USE_TARGET_ORIENTATION else 'OFF'}")
    print("  target_ori   = reset 직후 EE orientation을 사용해 전진 중 자세를 유지")

    # 주의: RMPFlowController는 생성 시점의 robot world pose를 base pose로 cache한다.
    # 그래서 반드시 initialize_robot()로 벽 부착 pose를 적용한 뒤 생성해야 한다.
    controller = RMPFlowController(
        name="m0609_fuel_port_wall_cylinder_rmpflow_controller",
        robot_articulation=robot,
        urdf_path=M0609_URDF_PATH,
        robot_description_path=M0609_DESCRIPTION_PATH,
        rmpflow_config_path=M0609_RMPFLOW_CONFIG_PATH,
        end_effector_frame_name="fuel_nozzle_tip", 
    )
    # 목표 위치를 주면 각 관절값을 계산해주는 두뇌

    print("  [OK] RMPFlowController 생성 완료")

    # [수정 포인트] 초기 고정 자세도 link_6가 아닌 실제 주유건 끝단의 영점 각도를 추출하여 고정합니다.
    ee_pos, locked_target_orientation = controller.rmp_flow.get_end_effector_pose(robot.get_joint_positions())
    locked_target_orientation = locked_target_orientation.copy()
    print(f"\n  EE 초기 위치          = {np.round(ee_pos, 4)}")
    print(f"  locked EE orientation = {np.round(locked_target_orientation, 4)}")
    print(f"  Fuel cylinder center = {FUEL_PORT_CENTER}")

    was_playing = False
    task_done = False
    step_count = 0

    while simulation_app.is_running():
        my_world.step(render=True)
        time.sleep(0.01)
        is_playing = my_world.is_playing()

        # Play 시작 감지 → 리셋
        if is_playing and not was_playing:
            my_world.reset()
            initialize_robot(robot, my_world)
            controller.reset()
            task.sequence.reset()
            
            # [수정 포인트] 리셋 시 방향 고정도 주유건 끝단 기준으로 가져옵니다.
            _, locked_target_orientation = controller.rmp_flow.get_end_effector_pose(robot.get_joint_positions())
            locked_target_orientation = locked_target_orientation.copy()
            
            task_done = False
            step_count = 0
            print("\n[RESET] 벽 부착형 축방향 직선 접근 sequence 시작")
            print(f"[RESET] locked EE orientation = {np.round(locked_target_orientation, 4)}\n")

        if is_playing and not task_done:
            obs = task.get_observations()
            
            # [수정 포인트] 로봇의 현재 상태 판별을 "link_6"가 아닌 "RMPflow가 계산한 진짜 주유건 끝단"의 위치/각도로 가져옵니다.
            ee_position, ee_orientation = controller.rmp_flow.get_end_effector_pose(
                obs["m0609_robot"]["joint_positions"]
            )

            stage = task.sequence.current

            # 08_return_home: RMPFlow 위치 제어가 아니라 joint 직접 보간 복귀
            if stage.name == "08_return_home":
                current_joints = robot.get_joint_positions()
                target_joints = build_initial_joint_positions(robot, current_joints)
                next_joints = current_joints + HOME_JOINT_SPEED_ALPHA * (target_joints - current_joints)
                # 천천히 이동 0.012 * (목표 - 현재)
                robot.set_joint_positions(next_joints)

                joint_err = np.linalg.norm(next_joints[:6] - target_joints[:6])
                if joint_err < HOME_JOINT_TOLERANCE:
                    task.sequence.hold_count += 1
                else:
                    task.sequence.hold_count = 0

                if step_count % PRINT_EVERY_N_STEPS == 0:
                    print(
                        f"[stage={task.sequence.index}:{stage.name}] "
                        f"joint_err={joint_err:.4f} "
                        f"hold={task.sequence.hold_count}/{stage.hold_steps}"
                    )

                if task.sequence.hold_count >= stage.hold_steps:
                    print("\n[STAGE END] 08_return_home -> home reached")
                    task.sequence.done = True
                    print("\n[완료] 벽 부착형 자동 주유 sequence 종료")
                    task_done = True
                    my_world.pause()

                step_count += 1
                was_playing = is_playing
                continue

            # 일반 waypoint stage: RMPFlow 위치 제어
            task_done = task.sequence.update(ee_position, ee_orientation)
            
            if task_done:
                print("\n[완료] 벽 부착형 자동 주유 waypoint sequence 종료")
                my_world.pause()
                was_playing = is_playing
                continue

            stage = task.sequence.current
            command_target = task.sequence.get_command_target(ee_position)

            if command_target is not None:
                # stage별 orientation 제어.
                if USE_TARGET_ORIENTATION and stage.use_orientation:
                    actions = controller.forward(
                        target_end_effector_position=command_target,
                        target_end_effector_orientation=locked_target_orientation,
                    )
                else:
                    actions = controller.forward(
                        target_end_effector_position=command_target,
                    )
                robot.apply_action(actions)

            step_count += 1
            if step_count % PRINT_EVERY_N_STEPS == 0:
                link_angle = obs["m0609_robot"].get("link5_to_link6_angle_deg")
                link_angle_str = "None" if link_angle is None else f"{link_angle:.2f}deg"
                print(task.sequence.debug_string(ee_position) + f" link5_to_link6_vs_insert={link_angle_str}")
                pass
        was_playing = is_playing

    simulation_app.close()


if __name__ == "__main__":
    main()