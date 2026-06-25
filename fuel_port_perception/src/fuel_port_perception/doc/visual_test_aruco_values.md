# visual_test.usda ArUco 기준값 요약

## 선택한 marker 설정
- Dictionary: DICT_4X4_50
- Marker ID: 0
- Marker pattern size: 0.12 m
- Marker world center: [-0.40267, -0.77000, 1.20000]
- Marker path: /World/aruco_vehicle_marker

## marker frame convention
- x = world +X
- y = world -Z, 즉 화면/마커 아래 방향
- z = world +Y, 즉 카메라/차체 바깥 방향

## target offsets for detector
- marker_to_door_xyz: [0.000000, 0.177070, -0.040710]
- marker_to_cap_xyz:  [-0.000340, 0.135590, -0.122680]
- marker_to_hole_xyz: [0.006870, 0.181809, -0.294154]

주의: marker_to_hole_xyz는 hole center가 아니라 mouth surface 기준이다.
현재 multi_robot_oiling 코드가 green 모드에서 apply_mouth_offset=True로 FUEL_PORT_DEPTH/2 만큼 안쪽 보정을 하기 때문이다.
