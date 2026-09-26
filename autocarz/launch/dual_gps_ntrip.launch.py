"""Bring up the ROS 1-compatible dual-GPS and NTRIP topology.

Both ZED-F9P rovers receive the correction stream from /rtcm.  The NTRIP
client obtains the VRS rover position from the rear receiver's /ublox1/nmea
topic, matching the ROS 1 launch configuration in autocarz0516.

If no ports are supplied, dual_ublox.launch.py selects the two currently
present /dev/ttyACM* devices.  This handles the ACM renumbering seen after a
USB reset; explicit rear_port/front_port arguments remain available when the
physical front/rear order must be overridden.

Credentials deliberately have no default.  Pass them as launch arguments;
do not add them to a launch file or shell history.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    autocarz_share = get_package_share_directory('autocarz')
    ntrip_share = get_package_share_directory('ntrip_client')

    dual_ublox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(autocarz_share, 'launch', 'dual_ublox.launch.py')
        ),
        launch_arguments={
            'rear_port': LaunchConfiguration('rear_port'),
            'front_port': LaunchConfiguration('front_port'),
            'rear_frame_id': LaunchConfiguration('rear_frame_id'),
            'front_frame_id': LaunchConfiguration('front_frame_id'),
        }.items(),
    )

    ntrip_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            # ntrip_client's setuptools data_files installs launch files
            # directly below its package share directory.
            os.path.join(ntrip_share, 'ntrip_client_launch.py')
        ),
        launch_arguments={
            'host': LaunchConfiguration('ntrip_host'),
            'port': LaunchConfiguration('ntrip_port'),
            'mountpoint': LaunchConfiguration('ntrip_mountpoint'),
            'authenticate': LaunchConfiguration('ntrip_authenticate'),
            'username': LaunchConfiguration('ntrip_username'),
            'password': LaunchConfiguration('ntrip_password'),
            'rtcm_message_package': 'rtcm_msgs',
            'nmea_topic': '/ublox1/nmea',
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'rear_port',
            default_value='',
            description='empty selects the first currently detected u-blox ACM port',
        ),
        DeclareLaunchArgument(
            'front_port',
            default_value='',
            description='empty selects the second currently detected u-blox ACM port',
        ),
        DeclareLaunchArgument('rear_frame_id', default_value='gps_back'),
        DeclareLaunchArgument('front_frame_id', default_value='gps_front'),
        DeclareLaunchArgument('ntrip_host', default_value='RTS2.ngii.go.kr'),
        DeclareLaunchArgument('ntrip_port', default_value='2101'),
        DeclareLaunchArgument('ntrip_mountpoint', default_value='VRS-RTCM32'),
        DeclareLaunchArgument('ntrip_authenticate', default_value='True'),
        DeclareLaunchArgument(
            'ntrip_username',
            default_value='',
            description='NTRIP account name (required when authenticate is True)',
        ),
        DeclareLaunchArgument(
            'ntrip_password',
            default_value='',
            description='NTRIP password (required when authenticate is True)',
        ),
        dual_ublox,
        ntrip_client,
    ])
