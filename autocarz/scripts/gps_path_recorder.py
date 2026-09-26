#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gps_path_recorder.py

======================================================================
1. 기능
======================================================================

ROS2 ISRO-P2 GPS의 /fix 토픽을 직접 받아 주행 Path를 저장한다.

/odom은 사용하지 않는다.

동작 순서:

    /fix
      ↓
    latitude / longitude
      ↓
    기존 sensor_data_isro.py와 동일한 latlong2xy()
      ↓
    x, y 평면좌표
      ↓
    Raw 데이터 저장
      ↓
    Ctrl+C
      ↓
    누적거리 기준 재샘플링
      ↓
    Path TXT 저장


======================================================================
2. 입력 Topic
======================================================================

Topic:

    /fix

Message:

    sensor_msgs/msg/NavSatFix


ISRO-P2 Driver가 실행되고 /fix가 들어오면
autocarz_vector.launch.py를 실행하지 않아도 Path를 기록할 수 있다.


======================================================================
3. 좌표 변환
======================================================================

기존 autocarz의:

    sensor_data_isro.py

에서 사용하는 latlong2xy()와 동일한 계산식을 사용한다.

중요:

    map_offset_x
    map_offset_y

는 실제 자율주행 시 사용하는 값과 동일해야 한다.


현재 autocarz_vector.launch.py 기본값:

    map_offset_x = -125320.0
    map_offset_y = 136746.0

현재 워크스페이스 코드 주석상 이 값은 UOS 경로용이다.

따라서 CBuniv 등 다른 시험장에서는
해당 시험장에서 사용하는 정확한 offset을 입력해야 한다.


======================================================================
4. 저장 폴더
======================================================================

장소 코드 + 본선/예선으로 자동 결정한다.

예:

    장소 = CBuniv
    본선

        ↓

    autocarz/path/CBuniv_born/


예:

    장소 = KNU
    예선

        ↓

    autocarz/path/KNU_yeah/


폴더가 없으면 자동으로 생성한다.


======================================================================
5. Path 저장 형식
======================================================================

    x y 0.000000 radius


예:

    123.123456789000 70.123456789000 0.000000 1.200000


======================================================================
6. 실행
======================================================================

GPS Driver:

    ros2 launch ISRO_P2_Driver ISRO_P2_Driver.launch.py mode:=serial

NTRIP 사용 시:

    ros2 run ISRO_P2_Driver ntrip.py


/fix 확인:

    ros2 topic hz /fix

또는:

    ros2 topic echo /fix --once


Path Recorder:

    ros2 run autocarz gps_path_recorder.py


======================================================================
7. 종료
======================================================================

주행이 끝나면:

    Ctrl + C

재샘플링 후 자동 저장한다.

======================================================================
"""

import csv
import math
import os
import sys
import threading

import rclpy

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import NavSatFix


# ======================================================================
# 1. 기본 설정
# ======================================================================

GPS_TOPIC = "/fix"

DEFAULT_PLACE = "CBuniv"

DEFAULT_ROUND = "본선"

DEFAULT_SAMPLING_SPACING = 0.3

DEFAULT_ROI_RADIUS = 1.2

DEFAULT_OUTPUT_FILENAME = "gps_path.txt"


# ----------------------------------------------------------------------
# 현재 autocarz_vector.launch.py의 기본값
#
# 코드 주석상 UOS 경로용 offset
# ----------------------------------------------------------------------

DEFAULT_MAP_OFFSET_X = -125320.0

DEFAULT_MAP_OFFSET_Y = 136746.0


# ======================================================================
# 2. 기본 Utility
# ======================================================================

def distance_2d(
    p1,
    p2
):
    """
    두 XY 좌표 사이 Euclidean distance.
    """

    dx = (
        p2[0]
        -
        p1[0]
    )

    dy = (
        p2[1]
        -
        p1[1]
    )

    return math.hypot(
        dx,
        dy
    )


def ask_string(
    message,
    default_value
):
    """
    문자열 입력.
    """

    value = input(
        f"{message} [{default_value}]: "
    ).strip()

    if value == "":
        return default_value

    return value


def ask_float(
    message,
    default_value
):
    """
    일반 실수 입력.

    offset은 음수가 가능하므로
    양수 제한을 하지 않는다.
    """

    while True:

        value = input(
            f"{message} [{default_value}]: "
        ).strip()

        if value == "":

            return float(
                default_value
            )

        try:

            result = float(
                value
            )

        except ValueError:

            print(
                "[ERROR] 숫자를 입력하십시오."
            )

            continue

        if not math.isfinite(
            result
        ):

            print(
                "[ERROR] 유효한 숫자를 입력하십시오."
            )

            continue

        return result


def ask_positive_float(
    message,
    default_value
):
    """
    0보다 큰 실수 입력.
    """

    while True:

        value = input(
            f"{message} [{default_value}]: "
        ).strip()

        if value == "":

            return float(
                default_value
            )

        try:

            result = float(
                value
            )

        except ValueError:

            print(
                "[ERROR] 숫자를 입력하십시오."
            )

            continue

        if (
            not math.isfinite(
                result
            )
            or
            result <= 0.0
        ):

            print(
                "[ERROR] 0보다 큰 숫자를 입력하십시오."
            )

            continue

        return result


# ======================================================================
# 3. 장소 입력
# ======================================================================

def ask_place():

    value = input(
        f"장소 코드 [{DEFAULT_PLACE}]: "
    ).strip()

    if value == "":
        value = DEFAULT_PLACE

    invalid_characters = '<>:"/\\|?*'

    if any(
        character in value
        for character in invalid_characters
    ):

        raise ValueError(
            "장소 코드에 사용할 수 없는 문자가 있습니다.\n"
            f"사용 불가 문자: {invalid_characters}"
        )

    return value


# ======================================================================
# 4. 본선 / 예선
# ======================================================================

def ask_round():

    while True:

        value = input(
            f"구분 (본선/예선) [{DEFAULT_ROUND}]: "
        ).strip()

        if value == "":
            value = DEFAULT_ROUND

        normalized = value.lower()

        # --------------------------------------------------------------
        # 본선
        # --------------------------------------------------------------

        if normalized in [
            "본선",
            "born",
            "final",
            "b"
        ]:

            return (
                "본선",
                "born"
            )

        # --------------------------------------------------------------
        # 예선
        # --------------------------------------------------------------

        if normalized in [
            "예선",
            "yeah",
            "qualifying",
            "qualifier",
            "y"
        ]:

            return (
                "예선",
                "yeah"
            )

        print(
            "[ERROR] '본선' 또는 '예선'을 입력하십시오."
        )


# ======================================================================
# 5. autocarz Source Package 위치
# ======================================================================

def find_package_source_directory():
    """
    일반 구조:

        ~/ros2_ws/src/autocarz/scripts/gps_path_recorder.py

    --symlink-install 환경에서도 실제 source script 위치를 우선 확인한다.
    """

    # ------------------------------------------------------------------
    # 실제 Python 파일 위치
    # ------------------------------------------------------------------

    real_script_path = os.path.realpath(
        __file__
    )

    script_directory = os.path.dirname(
        real_script_path
    )

    if os.path.basename(
        script_directory
    ) == "scripts":

        package_directory = os.path.dirname(
            script_directory
        )

        if os.path.isdir(
            package_directory
        ):

            return package_directory

    # ------------------------------------------------------------------
    # 현재 실행 위치 기준 탐색
    # ------------------------------------------------------------------

    current = os.path.abspath(
        os.getcwd()
    )

    while True:

        # 현재 위치 자체가 autocarz
        if (
            os.path.basename(
                current
            ) == "autocarz"
            and
            os.path.isdir(
                os.path.join(
                    current,
                    "scripts"
                )
            )
        ):

            return current

        # ros2_ws/src/autocarz 형태
        candidate = os.path.join(
            current,
            "src",
            "autocarz"
        )

        if os.path.isdir(
            candidate
        ):

            return candidate

        parent = os.path.dirname(
            current
        )

        if parent == current:
            break

        current = parent

    raise FileNotFoundError(
        "autocarz source package 위치를 찾지 못했습니다.\n\n"
        "예상 위치:\n"
        "~/ros2_ws/src/autocarz/\n\n"
        "colcon build 시 --symlink-install 사용을 권장합니다."
    )


# ======================================================================
# 6. 저장 폴더 생성
# ======================================================================

def create_event_path_directory(
    package_directory,
    place,
    round_suffix
):

    path_root = os.path.join(
        package_directory,
        "path"
    )

    os.makedirs(
        path_root,
        exist_ok=True
    )

    event_folder = (
        f"{place}_{round_suffix}"
    )

    event_directory = os.path.join(
        path_root,
        event_folder
    )

    existed_before = os.path.isdir(
        event_directory
    )

    os.makedirs(
        event_directory,
        exist_ok=True
    )

    return (
        os.path.abspath(
            event_directory
        ),
        existed_before
    )


# ======================================================================
# 7. GPS 위경도 -> 기존 autocarz 평면좌표
# ======================================================================

def latlong2xy(
    latitude,
    longitude,
    map_offset_x,
    map_offset_y
):
    """
    업로드된 autocarz의 sensor_data_isro.py에 있는
    latlong2xy() 계산식을 동일하게 사용한다.

    입력:

        latitude
        longitude

    출력:

        x
        y
    """

    RE = 6371.00877

    GRID = 5.0e-7

    SLAT1 = 30.0

    SLAT2 = 60.0

    OLON = 126.0

    OLAT = 38.0

    XO = 43.0

    YO = 136.0

    DEGRAD = (
        math.pi
        /
        180.0
    )

    re = (
        RE
        /
        GRID
    )

    slat1 = (
        SLAT1
        *
        DEGRAD
    )

    slat2 = (
        SLAT2
        *
        DEGRAD
    )

    olon = (
        OLON
        *
        DEGRAD
    )

    olat = (
        OLAT
        *
        DEGRAD
    )

    sn = (
        math.tan(
            math.pi
            *
            0.25
            +
            slat2
            *
            0.5
        )
        /
        math.tan(
            math.pi
            *
            0.25
            +
            slat1
            *
            0.5
        )
    )

    sn = (
        math.log(
            math.cos(
                slat1
            )
            /
            math.cos(
                slat2
            )
        )
        /
        math.log(
            sn
        )
    )

    sf = math.tan(
        math.pi
        *
        0.25
        +
        slat1
        *
        0.5
    )

    sf = (
        math.pow(
            sf,
            sn
        )
        *
        math.cos(
            slat1
        )
        /
        sn
    )

    ro = math.tan(
        math.pi
        *
        0.25
        +
        olat
        *
        0.5
    )

    ro = (
        re
        *
        sf
        /
        math.pow(
            ro,
            sn
        )
    )

    ra = math.tan(
        math.pi
        *
        0.25
        +
        latitude
        *
        DEGRAD
        *
        0.5
    )

    ra = (
        re
        *
        sf
        /
        math.pow(
            ra,
            sn
        )
    )

    theta = (
        longitude
        *
        DEGRAD
        -
        olon
    )

    if theta > math.pi:

        theta -= (
            2.0
            *
            math.pi
        )

    if theta < -math.pi:

        theta += (
            2.0
            *
            math.pi
        )

    theta *= sn

    rs_x = math.floor(
        ra
        *
        math.sin(
            theta
        )
        +
        XO
        +
        0.5
    )

    rs_y = math.floor(
        ro
        -
        ra
        *
        math.cos(
            theta
        )
        +
        YO
        +
        0.5
    )

    # ------------------------------------------------------------------
    # sensor_data_isro.py와 동일한 마지막 변환
    # ------------------------------------------------------------------

    rs_x = (
        rs_x
        /
        2000.0
        +
        map_offset_x
    )

    rs_y = (
        rs_y
        /
        2000.0
        +
        map_offset_y
    )

    return (
        float(
            rs_x
        ),
        float(
            rs_y
        )
    )


# ======================================================================
# 8. 연속 중복점 제거
# ======================================================================

def remove_consecutive_duplicates(
    points,
    tolerance=1.0e-6
):
    """
    points:

        [
            (x, y, z, timestamp),
            ...
        ]
    """

    if len(
        points
    ) == 0:

        return []

    cleaned = [
        points[0]
    ]

    for point in points[
        1:
    ]:

        if distance_2d(
            cleaned[-1],
            point
        ) > tolerance:

            cleaned.append(
                point
            )

    return cleaned


# ======================================================================
# 9. 누적거리 기준 Path 재샘플링
# ======================================================================

def resample_path(
    points,
    spacing
):

    points = remove_consecutive_duplicates(
        points
    )

    if len(
        points
    ) < 2:

        raise RuntimeError(
            "서로 다른 GPS 위치가 최소 2개 필요합니다."
        )

    # ------------------------------------------------------------------
    # 누적거리
    # ------------------------------------------------------------------

    cumulative_distance = [
        0.0
    ]

    for i in range(
        1,
        len(points)
    ):

        segment_distance = distance_2d(
            points[
                i - 1
            ],
            points[
                i
            ]
        )

        cumulative_distance.append(
            cumulative_distance[-1]
            +
            segment_distance
        )

    total_distance = cumulative_distance[
        -1
    ]

    if total_distance <= 0.0:

        raise RuntimeError(
            "전체 이동거리가 0입니다."
        )

    # ------------------------------------------------------------------
    # sampling target
    # ------------------------------------------------------------------

    target_distances = []

    target = 0.0

    while (
        target
        <=
        total_distance
        +
        1.0e-9
    ):

        target_distances.append(
            target
        )

        target += spacing

    # ------------------------------------------------------------------
    # 선형보간
    # ------------------------------------------------------------------

    sampled_points = []

    segment_index = 0

    for target in target_distances:

        while (
            segment_index
            <
            len(
                cumulative_distance
            )
            -
            2
            and
            cumulative_distance[
                segment_index + 1
            ]
            <
            target
        ):

            segment_index += 1

        s0 = cumulative_distance[
            segment_index
        ]

        s1 = cumulative_distance[
            segment_index + 1
        ]

        p0 = points[
            segment_index
        ]

        p1 = points[
            segment_index + 1
        ]

        segment_length = (
            s1
            -
            s0
        )

        if segment_length <= 1.0e-12:

            continue

        ratio = (
            target
            -
            s0
        ) / segment_length

        ratio = max(
            0.0,
            min(
                1.0,
                ratio
            )
        )

        x = (
            p0[0]
            +
            ratio
            *
            (
                p1[0]
                -
                p0[0]
            )
        )

        y = (
            p0[1]
            +
            ratio
            *
            (
                p1[1]
                -
                p0[1]
            )
        )

        sampled_points.append(
            (
                x,
                y
            )
        )

    return (
        sampled_points,
        total_distance
    )


# ======================================================================
# 10. ROS2 GPS Path Recorder
# ======================================================================

class GPSPathRecorder(Node):

    def __init__(
        self,
        sampling_spacing,
        roi_radius,
        map_offset_x,
        map_offset_y,
        path_output,
        raw_output
    ):

        super().__init__(
            "gps_path_recorder"
        )

        self.sampling_spacing = float(
            sampling_spacing
        )

        self.roi_radius = float(
            roi_radius
        )

        self.map_offset_x = float(
            map_offset_x
        )

        self.map_offset_y = float(
            map_offset_y
        )

        self.path_output = os.path.abspath(
            path_output
        )

        self.raw_output = os.path.abspath(
            raw_output
        )

        self.raw_points = []

        self.lock = threading.Lock()

        self.received_count = 0

        self.saved = False

        # --------------------------------------------------------------
        # /fix Subscriber
        #
        # sensor_data_isro.py와 동일하게 sensor_data QoS 사용
        # --------------------------------------------------------------

        self.subscription = self.create_subscription(
            NavSatFix,
            GPS_TOPIC,
            self.gps_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            "================================================"
        )

        self.get_logger().info(
            "GPS PATH RECORDER START"
        )

        self.get_logger().info(
            f"GPS topic      : {GPS_TOPIC}"
        )

        self.get_logger().info(
            f"Map offset X   : {self.map_offset_x}"
        )

        self.get_logger().info(
            f"Map offset Y   : {self.map_offset_y}"
        )

        self.get_logger().info(
            f"Sampling       : "
            f"{self.sampling_spacing:.3f} m"
        )

        self.get_logger().info(
            f"ROI radius     : "
            f"{self.roi_radius:.3f} m"
        )

        self.get_logger().info(
            f"Path output    : "
            f"{self.path_output}"
        )

        self.get_logger().info(
            "주행 종료 후 Ctrl+C를 누르십시오."
        )

        self.get_logger().info(
            "================================================"
        )


    # ==================================================================
    # GPS Callback
    # ==================================================================

    def gps_callback(
        self,
        msg
    ):

        latitude = float(
            msg.latitude
        )

        longitude = float(
            msg.longitude
        )

        altitude = float(
            msg.altitude
        )

        # --------------------------------------------------------------
        # 위경도 유효성 검사
        # --------------------------------------------------------------

        if (
            not math.isfinite(
                latitude
            )
            or
            not math.isfinite(
                longitude
            )
        ):

            self.get_logger().warning(
                "NaN/Inf GPS 좌표 제외"
            )

            return

        # --------------------------------------------------------------
        # 기존 자율주행 코드와 동일한 XY 변환
        # --------------------------------------------------------------

        try:

            (
                x,
                y
            ) = latlong2xy(
                latitude,
                longitude,
                self.map_offset_x,
                self.map_offset_y
            )

        except Exception as error:

            self.get_logger().warning(
                f"GPS 좌표 변환 실패: {error}"
            )

            return

        # --------------------------------------------------------------
        # altitude가 유효하지 않으면 0
        # --------------------------------------------------------------

        if not math.isfinite(
            altitude
        ):

            altitude = 0.0

        # --------------------------------------------------------------
        # ROS timestamp
        # --------------------------------------------------------------

        stamp = (
            float(
                msg.header.stamp.sec
            )
            +
            float(
                msg.header.stamp.nanosec
            )
            *
            1.0e-9
        )

        # --------------------------------------------------------------
        # Raw 저장
        # --------------------------------------------------------------

        with self.lock:

            self.raw_points.append(
                (
                    x,
                    y,
                    altitude,
                    stamp
                )
            )

            self.received_count += 1

            count = self.received_count

        # --------------------------------------------------------------
        # 처음 데이터는 즉시 출력
        # --------------------------------------------------------------

        if count == 1:

            self.get_logger().info(
                "첫 GPS 데이터 수신"
            )

            self.get_logger().info(
                f"lat/lon = "
                f"{latitude:.10f}, "
                f"{longitude:.10f}"
            )

            self.get_logger().info(
                f"x/y     = "
                f"{x:.6f}, "
                f"{y:.6f}"
            )

        # --------------------------------------------------------------
        # 100개마다 상태 확인
        # --------------------------------------------------------------

        if count % 100 == 0:

            self.get_logger().info(
                f"GPS 수신 데이터 : "
                f"{count} points / "
                f"x={x:.3f}, y={y:.3f}"
            )


    # ==================================================================
    # Raw CSV 저장
    # ==================================================================

    def save_raw_csv(
        self,
        points
    ):

        with open(
            self.raw_output,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow(
                [
                    "x",
                    "y",
                    "z",
                    "timestamp"
                ]
            )

            for (
                x,
                y,
                z,
                timestamp
            ) in points:

                writer.writerow(
                    [
                        f"{x:.12f}",
                        f"{y:.12f}",
                        f"{z:.12f}",
                        f"{timestamp:.9f}"
                    ]
                )


    # ==================================================================
    # 최종 Path 저장
    # ==================================================================

    def save_path_txt(
        self,
        sampled_points
    ):

        with open(
            self.path_output,
            "w",
            encoding="utf-8"
        ) as file:

            for (
                x,
                y
            ) in sampled_points:

                file.write(
                    f"{x:.12f} "
                    f"{y:.12f} "
                    f"0.000000 "
                    f"{self.roi_radius:.6f}\n"
                )


    # ==================================================================
    # 최종 저장
    # ==================================================================

    def save_all(
        self
    ):

        if self.saved:

            return

        self.saved = True

        with self.lock:

            points = list(
                self.raw_points
            )

        print()
        print(
            "================================================"
        )

        print(
            "PATH GENERATION"
        )

        print(
            "================================================"
        )

        print(
            f"/fix 수신 개수 : {len(points)}"
        )

        if len(
            points
        ) < 2:

            print(
                "[ERROR] /fix GPS 데이터가 2개 미만입니다."
            )

            print(
                "아래 명령으로 /fix 수신 여부를 확인하십시오:"
            )

            print(
                "ros2 topic hz /fix"
            )

            return

        # --------------------------------------------------------------
        # Raw 데이터 저장
        # --------------------------------------------------------------

        self.save_raw_csv(
            points
        )

        # --------------------------------------------------------------
        # 재샘플링
        # --------------------------------------------------------------

        try:

            (
                sampled_points,
                total_distance
            ) = resample_path(
                points,
                self.sampling_spacing
            )

        except RuntimeError as error:

            print(
                f"[ERROR] {error}"
            )

            return

        # --------------------------------------------------------------
        # 최종 Path 저장
        # --------------------------------------------------------------

        self.save_path_txt(
            sampled_points
        )

        # --------------------------------------------------------------
        # 결과
        # --------------------------------------------------------------

        print(
            f"Raw GPS 개수    : {len(points)}"
        )

        print(
            f"전체 주행거리   : "
            f"{total_distance:.3f} m"
        )

        print(
            f"Sampling 간격   : "
            f"{self.sampling_spacing:.3f} m"
        )

        print(
            f"Waypoint 개수   : "
            f"{len(sampled_points)}"
        )

        print(
            f"Map offset X    : "
            f"{self.map_offset_x}"
        )

        print(
            f"Map offset Y    : "
            f"{self.map_offset_y}"
        )

        print()

        print(
            "Raw CSV:"
        )

        print(
            self.raw_output
        )

        print()

        print(
            "Path TXT:"
        )

        print(
            self.path_output
        )

        print(
            "================================================"
        )


# ======================================================================
# 11. Main
# ======================================================================

def main(
    args=None
):

    print()
    print(
        "============================================================"
    )

    print(
        "ROS2 /fix GPS PATH RECORDER"
    )

    print(
        "============================================================"
    )

    # ------------------------------------------------------------------
    # autocarz source 위치
    # ------------------------------------------------------------------

    package_directory = (
        find_package_source_directory()
    )

    # ------------------------------------------------------------------
    # 장소
    # ------------------------------------------------------------------

    place = ask_place()

    # ------------------------------------------------------------------
    # 본선 / 예선
    # ------------------------------------------------------------------

    (
        round_name,
        round_suffix
    ) = ask_round()

    # ------------------------------------------------------------------
    # 폴더 생성
    # ------------------------------------------------------------------

    (
        event_directory,
        existed_before
    ) = create_event_path_directory(
        package_directory,
        place,
        round_suffix
    )

    print()

    if existed_before:

        print(
            "[INFO] 기존 Path 폴더 사용"
        )

    else:

        print(
            "[OK] 새 Path 폴더 생성"
        )

    print(
        event_directory
    )

    # ------------------------------------------------------------------
    # 좌표 offset
    # ------------------------------------------------------------------

    print()
    print(
        "---------------- 좌표계 설정 ----------------"
    )

    print(
        "주의: 기존 Path와 동일한 offset을 사용해야 합니다."
    )

    map_offset_x = ask_float(
        "Map offset X",
        DEFAULT_MAP_OFFSET_X
    )

    map_offset_y = ask_float(
        "Map offset Y",
        DEFAULT_MAP_OFFSET_Y
    )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    sampling_spacing = ask_positive_float(
        "Waypoint sampling 간격 [m]",
        DEFAULT_SAMPLING_SPACING
    )

    # ------------------------------------------------------------------
    # ROI radius
    # ------------------------------------------------------------------

    roi_radius = ask_positive_float(
        "ROI 반지름 [m]",
        DEFAULT_ROI_RADIUS
    )

    # ------------------------------------------------------------------
    # 파일 이름
    # ------------------------------------------------------------------

    output_filename = ask_string(
        "저장할 Path 파일명",
        DEFAULT_OUTPUT_FILENAME
    )

    if not output_filename.lower().endswith(
        ".txt"
    ):

        output_filename += ".txt"

    # ------------------------------------------------------------------
    # 저장 파일 위치
    # ------------------------------------------------------------------

    path_output = os.path.join(
        event_directory,
        output_filename
    )

    base_filename = os.path.splitext(
        output_filename
    )[0]

    raw_output = os.path.join(
        event_directory,
        base_filename
        +
        "_raw.csv"
    )

    # ------------------------------------------------------------------
    # 설정 출력
    # ------------------------------------------------------------------

    print()
    print(
        "---------------- 설정 ----------------"
    )

    print(
        f"장소           : {place}"
    )

    print(
        f"구분           : {round_name}"
    )

    print(
        f"GPS Topic      : {GPS_TOPIC}"
    )

    print(
        f"Map offset X   : {map_offset_x}"
    )

    print(
        f"Map offset Y   : {map_offset_y}"
    )

    print(
        f"Sampling 간격  : "
        f"{sampling_spacing:.3f} m"
    )

    print(
        f"ROI radius     : "
        f"{roi_radius:.3f} m"
    )

    print(
        f"저장 폴더      : "
        f"{event_directory}"
    )

    print(
        f"Path 파일      : "
        f"{output_filename}"
    )

    print(
        "--------------------------------------"
    )

    print()

    # ------------------------------------------------------------------
    # ROS2 초기화
    # ------------------------------------------------------------------

    rclpy.init(
        args=args
    )

    node = GPSPathRecorder(
        sampling_spacing=sampling_spacing,
        roi_radius=roi_radius,
        map_offset_x=map_offset_x,
        map_offset_y=map_offset_y,
        path_output=path_output,
        raw_output=raw_output
    )

    try:

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        print()
        print(
            "[INFO] Ctrl+C 감지"
        )

    finally:

        node.save_all()

        node.destroy_node()

        if rclpy.ok():

            rclpy.shutdown()


# ======================================================================
# Entry Point
# ======================================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print(
            f"[ERROR] {error}",
            file=sys.stderr
        )

        sys.exit(
            1
        )