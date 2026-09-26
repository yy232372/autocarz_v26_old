# 본선 Track ROI + Path-to-ROI mapping 전용 launch
# 예선 LiDAR 설정은 포함하지 않는다.
# 기존 autocarz_lidar.launch.py와 독립적으로 사용한다.
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

import glob
import os


# =============================================================================
# 트랙 데이터 폴더 설정
# =============================================================================
#
# 파일명 뒤 수식어(best, middle, v2 등)는 상관없다.
# launch에서는 폴더만 지정하고 아래 prefix를 가진 파일을 자동 검색한다.
#
#   path_in*.csv 또는 path_in*.txt
#   path_out*.csv 또는 path_out*.txt
#   in_dense*.csv
#   out_dense*.csv
#   in_path_roi_map*.csv
#   out_path_roi_map*.csv
#
# 같은 종류의 파일이 2개 이상 발견되면 임의 선택하지 않고 오류로 중단한다.
# 기준 루트는 path/path_CBBorn 으로 고정한다.
# 사용자는 그 아래의 폴더 이름(best, middle, first 등)만 선택한다.
#
TRACK_ROOT = 'path/path_CBBorn'
TRACK_FOLDER = 'best'


FILE_PATTERNS = {
    'driving_path_in': ('path_in*.csv', 'path_in*.txt'),
    'driving_path_out': ('path_out*.csv', 'path_out*.txt'),
    'roi_in': ('in_dense*.csv',),
    'roi_out': ('out_dense*.csv',),
    'mapping_in': ('in_path_roi_map*.csv',),
    'mapping_out': ('out_path_roi_map*.csv',),
}


def find_unique_file(search_dir, label, patterns):
    matches = []
    for pattern in patterns:
        matches.extend(glob.glob(os.path.join(search_dir, pattern)))

    matches = sorted(
        {os.path.abspath(path) for path in matches if os.path.isfile(path)}
    )

    if len(matches) == 0:
        pattern_text = ', '.join(patterns)
        raise RuntimeError(
            f'[autocarz_lidar_track] {label} 파일을 찾을 수 없습니다. '
            f'폴더: {search_dir} / 패턴: {pattern_text}'
        )

    if len(matches) > 1:
        file_list = '\n  - '.join(matches)
        raise RuntimeError(
            f'[autocarz_lidar_track] {label} 후보가 {len(matches)}개 발견되었습니다. '
            f'같은 종류의 파일은 폴더에 1개만 두십시오.\n  - {file_list}'
        )

    return matches[0]


def launch_setup(context, *args, **kwargs):
    carla = LaunchConfiguration('carla').perform(context).strip().lower()
    track_folder = LaunchConfiguration('track_folder').perform(context).strip()

    package_share = FindPackageShare('autocarz').perform(context)

    # track_folder에는 best / middle / first 같은 하위 폴더명만 허용한다.
    # 경로 전체를 넘기지 않게 해서 기준 루트가 바뀌는 실수를 막는다.
    if (
        not track_folder
        or track_folder in ('.', '..')
        or '/' in track_folder
        or '\\' in track_folder
    ):
        raise RuntimeError(
            '[autocarz_lidar_track] track_folder에는 하위 폴더 이름만 입력하십시오. '
            '예: best, middle, first'
        )

    search_dir = os.path.normpath(
        os.path.join(package_share, TRACK_ROOT, track_folder)
    )

    if not os.path.isdir(search_dir):
        raise RuntimeError(
            '[autocarz_lidar_track] 트랙 데이터 폴더가 없습니다: '
            + search_dir
        )

    driving_path_in = find_unique_file(
        search_dir, 'IN driving path', FILE_PATTERNS['driving_path_in']
    )
    driving_path_out = find_unique_file(
        search_dir, 'OUT driving path', FILE_PATTERNS['driving_path_out']
    )
    roi_in = find_unique_file(
        search_dir, 'IN Dense ROI', FILE_PATTERNS['roi_in']
    )
    roi_out = find_unique_file(
        search_dir, 'OUT Dense ROI', FILE_PATTERNS['roi_out']
    )
    mapping_in = find_unique_file(
        search_dir, 'IN Path-to-ROI mapping', FILE_PATTERNS['mapping_in']
    )
    mapping_out = find_unique_file(
        search_dir, 'OUT Path-to-ROI mapping', FILE_PATTERNS['mapping_out']
    )

    print('[autocarz_lidar_track] TRACK DIR :', search_dir)
    print('[autocarz_lidar_track] path_in   :', driving_path_in)
    print('[autocarz_lidar_track] path_out  :', driving_path_out)
    print('[autocarz_lidar_track] roi_in    :', roi_in)
    print('[autocarz_lidar_track] roi_out   :', roi_out)
    print('[autocarz_lidar_track] map_in    :', mapping_in)
    print('[autocarz_lidar_track] map_out   :', mapping_out)

    use_carla = carla in (
        'true',
        '1',
        'yes',
        'on',
    )

    if use_carla:
        vlp_pointcloud_topic = (
            '/carla/ego_vehicle/lidar_vlp16'
        )
        livox_pointcloud_topic = (
            '/carla/ego_vehicle/lidar_vlp16'
        )
    else:
        vlp_pointcloud_topic = '/velodyne_points'
        livox_pointcloud_topic = '/livox/lidar'

    nodes = []

    # =========================================================================
    # VLP-16 파이프라인
    #
    # /velodyne_points
    #   ├─ /vlp/roi_in
    #   │    └─ /vlp/ransac_in
    #   │          └─ /vlp/cluster_in
    #   └─ /vlp/roi_out
    #        └─ /vlp/ransac_out
    #              └─ /vlp/cluster_out
    # =========================================================================

    nodes.append(
        Node(
            package='autocarz',
            executable='roi_in_track',
            name='roi_in_track',
            namespace='vlp',
            output='screen',
            parameters=[
                {
                    'driving_path': driving_path_in,
                    'roi_in': roi_in,
                    'mapping_in': mapping_in,

                    'topics.pointcloud':
                        vlp_pointcloud_topic,
                    'topics.odom':
                        '/odom',
                    'topics.waypoint_info':
                        '/waypointInfo',

                    'topics.roi':
                        '/vlp/roi_in',
                    'topics.max_roi':
                        '/vlp/max_roi_dis_in',

                    'roi_idxlen': 54,
                    'roi_idx_start': 7,
                    'is_livox': False,
                    'radius_dim': 0.1,
                }
            ],
        )
    )

    nodes.append(
        Node(
            package='autocarz',
            executable='roi_out_track',
            name='roi_out_track',
            namespace='vlp',
            output='screen',
            parameters=[
                {
                    'driving_path': driving_path_out,
                    'roi_out': roi_out,
                    'mapping_out': mapping_out,

                    'topics.pointcloud':
                        vlp_pointcloud_topic,
                    'topics.odom':
                        '/odom',
                    'topics.waypoint_info':
                        '/waypointInfo',

                    'topics.roi':
                        '/vlp/roi_out',
                    'topics.max_roi':
                        '/vlp/max_roi_dis_out',

                    'roi_idxlen': 54,
                    'roi_idx_start': 7,
                    'is_livox': False,
                    'radius_dim': 0.2,
                }
            ],
        )
    )

    nodes.append(
        Node(
            package='autocarz',
            executable='ransac_in',
            name='ransac_in',
            namespace='vlp',
            output='screen',
            parameters=[
                {
                    'z_threshold': 0.1,
                    'topics.input':
                        '/vlp/roi_in',
                    'topics.output':
                        '/vlp/ransac_in',
                }
            ],
        )
    )

    nodes.append(
        Node(
            package='autocarz',
            executable='ransac_out',
            name='ransac_out',
            namespace='vlp',
            output='screen',
            parameters=[
                {
                    'z_threshold': 0.1,
                    'topics.input':
                        '/vlp/roi_out',
                    'topics.output':
                        '/vlp/ransac_out',
                }
            ],
        )
    )

    nodes.append(
        Node(
            package='autocarz',
            executable='cluster_in',
            name='cluster_in',
            namespace='vlp',
            output='screen',
            parameters=[
                {
                    'topics.input':
                        '/vlp/ransac_in',
                    'topics.waypoint_info':
                        '/waypointInfo',

                    'topics.distance_front':
                        '/vlp/distance_front',
                    'topics.distance_side_front':
                        '/vlp/distance_side_front',
                    'topics.distance_side_back':
                        '/vlp/distance_side_back',

                    'topics.cluster':
                        '/vlp/cluster_in',
                }
            ],
        )
    )

    nodes.append(
        Node(
            package='autocarz',
            executable='cluster_out',
            name='cluster_out',
            namespace='vlp',
            output='screen',
            parameters=[
                {
                    'topics.input':
                        '/vlp/ransac_out',
                    'topics.waypoint_info':
                        '/waypointInfo',

                    'topics.distance_front':
                        '/vlp/distance_front',
                    'topics.distance_side_front':
                        '/vlp/distance_side_front',
                    'topics.distance_side_back':
                        '/vlp/distance_side_back',

                    'topics.cluster':
                        '/vlp/cluster_out',
                }
            ],
        )
    )

    # =========================================================================
    # Livox 파이프라인
    #
    # /livox/lidar
    #   ├─ /livox/roi_in
    #   │    └─ /livox/ransac_in
    #   │          └─ /livox/cluster_in
    #   └─ /livox/roi_out
    #        └─ /livox/ransac_out
    #              └─ /livox/cluster_out
    # =========================================================================

    nodes.append(
        Node(
            package='autocarz',
            executable='roi_in_track',
            name='roi_in_track',
            namespace='livox',
            output='screen',
            parameters=[
                {
                    'driving_path': driving_path_in,
                    'roi_in': roi_in,
                    'mapping_in': mapping_in,

                    'topics.pointcloud':
                        livox_pointcloud_topic,
                    'topics.odom':
                        '/odom',
                    'topics.waypoint_info':
                        '/waypointInfo',

                    'topics.roi':
                        '/livox/roi_in',
                    'topics.max_roi':
                        '/livox/max_roi_dis_in',

                    'roi_idxlen': 100,
                    'roi_idx_start': 7,
                    'is_livox': True,
                    'radius_dim': 0.2,
                }
            ],
        )
    )

    nodes.append(
        Node(
            package='autocarz',
            executable='roi_out_track',
            name='roi_out_track',
            namespace='livox',
            output='screen',
            parameters=[
                {
                    'driving_path': driving_path_out,
                    'roi_out': roi_out,
                    'mapping_out': mapping_out,

                    'topics.pointcloud':
                        livox_pointcloud_topic,
                    'topics.odom':
                        '/odom',
                    'topics.waypoint_info':
                        '/waypointInfo',

                    'topics.roi':
                        '/livox/roi_out',
                    'topics.max_roi':
                        '/livox/max_roi_dis_out',

                    'roi_idxlen': 100,
                    'roi_idx_start': 7,
                    'is_livox': True,
                    'radius_dim': 0.3,
                }
            ],
        )
    )

    nodes.append(
        Node(
            package='autocarz',
            executable='ransac_in',
            name='ransac_in',
            namespace='livox',
            output='screen',
            parameters=[
                {
                    'z_threshold': 0.25,
                    'topics.input':
                        '/livox/roi_in',
                    'topics.output':
                        '/livox/ransac_in',
                }
            ],
        )
    )

    nodes.append(
        Node(
            package='autocarz',
            executable='ransac_out',
            name='ransac_out',
            namespace='livox',
            output='screen',
            parameters=[
                {
                    'z_threshold': 0.25,
                    'topics.input':
                        '/livox/roi_out',
                    'topics.output':
                        '/livox/ransac_out',
                }
            ],
        )
    )

    nodes.append(
        Node(
            package='autocarz',
            executable='cluster_in',
            name='cluster_in',
            namespace='livox',
            output='screen',
            parameters=[
                {
                    'topics.input':
                        '/livox/ransac_in',
                    'topics.waypoint_info':
                        '/waypointInfo',

                    'topics.distance_front':
                        '/livox/distance_front',
                    'topics.distance_side_front':
                        '/livox/distance_side_front',
                    'topics.distance_side_back':
                        '/livox/distance_side_back',

                    'topics.cluster':
                        '/livox/cluster_in',
                }
            ],
        )
    )

    nodes.append(
        Node(
            package='autocarz',
            executable='cluster_out',
            name='cluster_out',
            namespace='livox',
            output='screen',
            parameters=[
                {
                    'topics.input':
                        '/livox/ransac_out',
                    'topics.waypoint_info':
                        '/waypointInfo',

                    'topics.distance_front':
                        '/livox/distance_front',
                    'topics.distance_side_front':
                        '/livox/distance_side_front',
                    'topics.distance_side_back':
                        '/livox/distance_side_back',

                    'topics.cluster':
                        '/livox/cluster_out',
                }
            ],
        )
    )

    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'carla',
                default_value='false',
                description='use Carla PointCloud2 topic',
            ),

            # -----------------------------------------------------------------
            # path/path_CBBorn 아래에서 사용할 폴더 이름만 지정한다.
            #
            # 예:
            #   path/path_CBBorn/best
            #   path/path_CBBorn/middle
            #   path/path_CBBorn/first
            #
            # 실행:
            # ros2 launch autocarz autocarz_lidar_track.launch.py \
            #     track_folder:=middle
            #
            # 선택된 폴더 안에서 path_in*, path_out*, in_dense*,
            # out_dense*, in_path_roi_map*, out_path_roi_map*를 자동 검색한다.
            # -----------------------------------------------------------------
            DeclareLaunchArgument(
                'track_folder',
                default_value=TRACK_FOLDER,
                description='path/path_CBBorn 아래의 트랙 세트 폴더명',
            ),

            OpaqueFunction(function=launch_setup),
        ]
    )
