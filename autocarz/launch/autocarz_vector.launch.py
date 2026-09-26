from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    mode = LaunchConfiguration('mode').perform(context).strip().lower()
    path_in_override = LaunchConfiguration('path_in').perform(context).strip()
    path_out_override = LaunchConfiguration('path_out').perform(context).strip()

    if mode == 'final':
        mode_name = 'final'
        path_in = 'path/CBuniv_born/PP_in_26.txt'
        path_out = 'path/CBuniv_born/PP_out_26.txt'
        velocity_scale_base = '{"0": 20.0, "1": 18.0, "2": 18.0, "3": 20.0, "4": 20.0}'
    elif mode == 'pre':
        mode_name = 'pre'
        path_in = 'path/CBuniv_yeah/path_yeah_opt_v4_labeled.txt'
        path_out = 'path/CBuniv_yeah/path_yeah_opt_v4_labeled.txt'
        velocity_scale_base = '{"0": 20.0, "1": 20.0, "2": 20.0, "3": 20.0}'
    else:
        raise RuntimeError(
            f"지원하지 않는 mode입니다: {mode}. "
            "final 또는 pre를 사용하세요."
        )
    if path_in_override:
        path_in = path_in_override
    if path_out_override:
        path_out = path_out_override

    planner_vector_node = Node(
        package='autocarz',
        executable='main.py',
        name='planner_vector',
        output='screen',
        parameters=[
            {
                'mode.name': mode_name,
                'mode.pathIn': path_in,
                'mode.pathOut': path_out,
                'mode.velocity_scale_base': velocity_scale_base,
            },
        ],
    )

    serial_control_node = Node(
        package='autocarz',
        executable='serialControl.py',
        name='serialControl',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_serial')),
        parameters=[
            {
                'port': LaunchConfiguration('serial_port'),
                'baud': LaunchConfiguration('serial_baud'),
            },
        ],
    )

    rqt_node = Node(
        package='autocarz',
        executable='ERP42RQT.py',
        name='erp42_rqt_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_rqt')),
        parameters=[{'mode.name': mode_name}],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz_lidarm',
        arguments=[
            '-d',
            f'{get_package_share_directory("autocarz")}/rviz/autocarz.rviz',
        ],
        output='screen',
    )

    odom_map_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='link1_broadcaster',
        arguments=[
            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '1',
            'odom',
            'map',
        ],
        output='screen',
    )

    base_velodyne_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='link2_broadcaster',
        arguments=[
            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '1',
            'base_link',
            'velodyne',
        ],
        output='screen',
    )

    # Keep the legacy Livox/VLP visualization convention: the two LiDAR
    # frames use the same origin and orientation.  This is the identity
    # mounting configuration used by the previous Livox-Velodyne RViz launch.
    # The parent is base_link here so it does not create a second parent for
    # the existing velodyne frame.
    base_livox_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='link3_broadcaster',
        arguments=[
            '0',
            '0',
            '0',
            '0',
            '0',
            '0',
            '1',
            'base_link',
            'livox_frame',
        ],
        output='screen',
    )

    return [
        planner_vector_node,
        serial_control_node,
        rqt_node,
        rviz_node,
        odom_map_tf_node,
        base_velodyne_tf_node,
        base_livox_tf_node,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'mode',
                default_value='final',
                description=(
                    'competition mode: final or pre'
                ),
            ),

            DeclareLaunchArgument(
                'path_in',
                default_value='',
                description=(
                    'optional package-relative/absolute test path override'
                ),
            ),

            DeclareLaunchArgument(
                'path_out',
                default_value='',
                description=(
                    'optional package-relative/absolute test path override'
                ),
            ),

            DeclareLaunchArgument(
                'use_serial',
                default_value='true',
                description=(
                    'start the real ERP42 serial-control node '
                    '(set false only for non-vehicle tests)'
                ),
            ),

            DeclareLaunchArgument(
                'use_rqt',
                default_value='false',
                description='start the ERP42RQT monitor with the selected mode',
            ),

            DeclareLaunchArgument(
                'serial_port',
                default_value='/dev/ttyUSB0',
                description='ERP42 serial device',
            ),

            DeclareLaunchArgument(
                'serial_baud',
                default_value='115200',
                description='ERP42 serial baud rate',
            ),

            OpaqueFunction(
                function=launch_setup,
            ),
        ]
    )
