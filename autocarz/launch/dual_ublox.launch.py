"""Start the two rover GPS receivers expected by the Autocarz controller.

The node names intentionally match the controller's fixed subscriptions and
the ROS 1 configuration:
  /ublox1/fix  (rear receiver)
  /ublox2/fix  (front receiver)

For an NGII VRS NTRIP client, launch ntrip_client with
``nmea_topic:=/ublox1/nmea`` so it can send the rover's GGA sentence to the
caster.  The client publishes the received correction stream on /rtcm, which
both ROS 2 u-blox nodes subscribe to.
"""

import glob
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _ublox_node(name: str, device: LaunchConfiguration,
                frame_id: LaunchConfiguration, config_path: str) -> Node:
    return Node(
        package='ublox_gps',
        executable='ublox_gps_node',
        name=name,
        output='screen',
        parameters=[
            config_path,
            {
                'device': device,
                'frame_id': frame_id,
            },
        ],
        # The ROS 2 u-blox driver creates its NMEA publisher as the relative
        # topic "nmea" (not the private topic "~/nmea").  Give each receiver
        # the ROS 1-compatible per-node topic explicitly.
        remappings=[
            ('nmea', f'/{name}/nmea'),
        ],
    )


def launch_setup(context, *args, **kwargs):
    config_path = os.path.join(
        get_package_share_directory('autocarz'),
        'config',
        'zed_f9p_dual.yaml',
    )

    requested_rear = LaunchConfiguration('rear_port').perform(context).strip()
    requested_front = LaunchConfiguration('front_port').perform(context).strip()
    if not requested_rear and not requested_front:
        # ttyACM numbering can shift after a USB reset.  Select the two
        # currently present ports at launch time rather than assuming that
        # the previous enumeration (ACM1/ACM2) still exists.
        ports = sorted(glob.glob('/dev/ttyACM*'))
        if len(ports) < 2:
            raise RuntimeError(
                'u-blox GPS 포트가 2개 필요하지만 현재 다음만 발견했습니다: '
                + ', '.join(ports)
            )
        requested_rear, requested_front = ports[:2]
    elif not requested_rear or not requested_front:
        raise RuntimeError(
            'rear_port와 front_port는 둘 다 지정하거나 둘 다 비워야 합니다.'
        )

    return [
        _ublox_node(
            'ublox1',
            requested_rear,
            LaunchConfiguration('rear_frame_id'),
            config_path,
        ),
        _ublox_node(
            'ublox2',
            requested_front,
            LaunchConfiguration('front_frame_id'),
            config_path,
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            'rear_port',
            default_value='',
            description=(
                'rear u-blox port; empty selects the first detected ACM port '
                'and publishes as /ublox1/fix'
            ),
        ),
        DeclareLaunchArgument(
            'front_port',
            default_value='',
            description=(
                'front u-blox port; empty selects the second detected ACM port '
                'and publishes as /ublox2/fix'
            ),
        ),
        DeclareLaunchArgument(
            'rear_frame_id',
            default_value='gps_back',
            description='frame ID for the rear receiver',
        ),
        DeclareLaunchArgument(
            'front_frame_id',
            default_value='gps_front',
            description='frame ID for the front receiver',
        ),
        OpaqueFunction(function=launch_setup),
    ])
