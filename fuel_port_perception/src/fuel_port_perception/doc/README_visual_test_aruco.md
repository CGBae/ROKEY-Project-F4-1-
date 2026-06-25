# visual_test.usda용 ArUco 인식 패키지

이 패키지는 `visual_test.usda`의 실제 차량/주유소 USD 구도에 맞춰 다시 만든 ArUco 버전입니다.
기존 색상 detector 대신 실행하며, 기존 Isaac Sim 메인 코드가 받는 `/color_detector/*` 토픽 이름은 그대로 유지합니다.

## 파일

- `aruco_marker_detector_visual_test.py`
  - 기존 `multi_color_detector.py`와 같은 ROS2 인터페이스를 유지하는 ArUco detector
  - `/rgb`, `/camera_info`, `/color_detector/mode_switch` 구독
  - `/color_detector/pose`, `/color_detector/target_locked`, `/color_detector/current_mode`, `/color_detector/debug_image` 발행

- `create_aruco_marker_grid_visual_test.py`
  - Isaac Sim Script Editor에서 실행
  - `/World/aruco_vehicle_marker`에 4x4_50 id=0 marker를 geometry cell로 생성
  - texture 파일이 필요 없음

- `aruco_detector_params_visual_test.yaml`
  - visual_test.usda 기준 offset과 marker size가 들어간 params

- `visual_test_aruco_values.md`
  - marker 위치와 offset 계산값 요약

## 적용 순서

### 1. Isaac Sim에서 marker 생성

`visual_test.usda`를 연 뒤, Script Editor에서 `create_aruco_marker_grid_visual_test.py`를 실행합니다.

생성 위치:

```text
/World/aruco_vehicle_marker
```

기본 marker world center:

```text
[-0.40267, -0.77000, 1.20000]
```

`rqt_image_view`에서 `/rgb`를 보고 marker가 보이는지 확인합니다.
안 보이면 `MARKER_CENTER_WORLD`를 조정하세요.

### 2. ROS2 패키지에 detector 복사

예시:

```bash
cp aruco_marker_detector_visual_test.py ~/fuel_ws/src/fuel_port_perception/fuel_port_perception/aruco_marker_detector_visual_test.py
```

### 3. setup.py entry point 추가

`setup.py`의 `console_scripts`에 추가:

```python
'aruco_marker_detector_visual_test = fuel_port_perception.aruco_marker_detector_visual_test:main',
```

빌드:

```bash
cd ~/fuel_ws
colcon build --packages-select fuel_port_perception
source install/setup.bash
```

### 4. 실행

기존 `multi_color_detector.py`는 끄고, ArUco detector만 실행하세요. 둘 다 같은 `/color_detector/pose`를 발행하므로 동시에 켜면 안 됩니다.

```bash
ros2 run fuel_port_perception aruco_marker_detector_visual_test \
  --ros-args --params-file /path/to/aruco_detector_params_visual_test.yaml
```

확인:

```bash
ros2 topic echo /color_detector/target_locked
ros2 topic echo /color_detector/pose
rqt_image_view   # /color_detector/debug_image 선택
```

## mode별 의미

- `blue`: marker 기준 fuel_cap 위치 발행
- `green`: marker 기준 fuel_port_hole mouth surface 위치 발행
- `yellow`: marker 기준 fuel_door 위치 발행

`green`에서 mouth surface를 발행하는 이유는 메인 코드가 `apply_mouth_offset=True`로 주유구 중심 보정을 한 번 더 하기 때문입니다.

## 발표용 설명 문장

초기에는 색상 기반 contour로 마개와 주유구를 구분했지만 실제 차량 환경에서는 색상만으로 주유구를 안정적으로 구분하기 어렵다. 이를 보완하기 위해 차량 주유구 주변에 ArUco marker를 배치하고, marker pose를 기준 좌표계로 사용하여 마개와 주유구 입구의 상대 위치를 계산하도록 확장하였다. 기존 ROS2 인터페이스는 유지하여 로봇 제어부 수정은 최소화하였다.
