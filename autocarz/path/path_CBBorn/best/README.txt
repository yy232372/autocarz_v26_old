본선 트랙 세트 — best (safety 1.3)
생성 2026-09-20

[ 배치 ]
  autocarz 소스 트리의  path/path_CBBorn/best/  에 이 6개 파일을 넣고 colcon build.
  CMakeLists.txt 가 launch 와 path 디렉토리를 share/autocarz 로 install 합니다.
  install/ 폴더에 직접 넣으면 다음 빌드 때 지워집니다.

  실행:  ros2 launch autocarz autocarz_lidar_track.launch.py track_folder:=best

[ 파일 ]
  path_in_best.txt           IN  주행경로  1423점 426.6 m   x y 구간라벨 ROI반지름 (탭 4열)
  path_out_best.txt          OUT 주행경로  1498점 449.1 m
  in_dense_best.csv          IN  dense ROI  1469점   x,y,roi_radius,distance_to_left,distance_to_middle1
  out_dense_best.csv         OUT dense ROI  1539점
  in_path_roi_map_best.csv   IN  path_index -> roi_index 매핑
  out_path_roi_map_best.csv  OUT 매핑

[ 주의 ]
  * 같은 종류 파일을 2개 두지 마십시오.
    런치가 path_in*.csv 와 path_in*.txt 를 함께 검색해 2개 이상이면 오류로 중단합니다.
    .csv 와 .txt 를 같이 넣으면 바로 이 경우에 걸립니다.

  * 경로를 바꾸면 매핑도 다시 만들어야 합니다.
    매핑은 경로의 점마다 한 줄씩 들어있고 roi_index 가 그 점 기준 최근접 dense 점으로
    박혀 있습니다. 로더가 path_index 0..n-1 연속성과 중복을 검사하므로, 점 개수가
    달라지면 노드가 뜨지 않습니다.
    dense 는 차선만으로 만들어지므로 경로가 바뀌어도 재사용할 수 있습니다.

  * ROI 반경은 dense 파일에서 옵니다.
    roi_in_track.cpp:367  effective_radius = roi_radius[roi_index] - radius_dim
    경로 파일 4열은 이 노드가 읽지 않습니다 (제어기 참고용으로 채워둠).
    dense 점은 차선 중앙선 위에 있고 반경은 차선폭의 절반이라, 주행선을 어디로 잡든
    ROI 원이 차선을 꽉 채웁니다.

  * 예선 코드와 달리 radius <= 0 으로 ROI 를 끄는 규약이 없습니다.
    이 버전은 effective_radius 를 그대로 제곱해 쓰므로 음수를 넣어도 꺼지지 않습니다.

[ 경로 제원 ]
                  IN              OUT
  길이            426.6 m         449.1 m
  점 간격         0.30 m 등간격   0.30 m 등간격
  벽 여유 최소    1.14 m          1.26 m       (차폭 절반 0.59 m)
  안쪽 벽까지     중앙 1.33 m     중앙 1.32 m
  곡률 최대       0.251 1/m       0.172        (한계 0.511, 최소회전반경 1.96 m)
  조향속도 p99    16.0 deg/s      10.1
  dense ROI       중앙 1.60 m     중앙 1.58 m
  구간 라벨       14개            14개          0 직선 / 1 곡선

  IN  코리도: left <-> middle1   (로터리에서 화단 남쪽)
  OUT 코리도: middle2 <-> right  (로터리에서 화단 북쪽)

  랩타임은 모델 추정치입니다 (횡가속도 8.0 m/s^2, v_max 20 km/h, 조향속도 20 deg/s 가정).
  실측이 아닙니다.

[ 라벨 ]
  0 직선 · 1 곡선 · 2 곡선 · 3 미사용 · 4 합류
  현재 0 과 1 만 사용했습니다. 차선변경 지점에 4 가 필요하면 path_edit 로 추가하십시오.
