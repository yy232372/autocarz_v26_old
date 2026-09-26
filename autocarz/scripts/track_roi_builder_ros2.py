#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


EPS = 1.0e-9


# ============================================================
# Default Path
# ============================================================
#
# 이 파일이:
#   autocarz/src/track_roi_builder.py
#
# 에 있다고 가정하면 자동으로:
#   autocarz/path/track_data
#
# 를 사용한다.
#
SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PACKAGE_DIR = os.path.dirname(
    SCRIPT_DIR
)

DEFAULT_DATA_DIR = os.path.join(
    PACKAGE_DIR,
    "path",
    "track_data",
)

DEFAULT_JSON_PATH = os.path.join(
    DEFAULT_DATA_DIR,
    "track_roi.json",
)

DEFAULT_TXT_PATH = os.path.join(
    DEFAULT_DATA_DIR,
    "track_roi.txt",
)

DEFAULT_FIGURE_PATH = os.path.join(
    DEFAULT_DATA_DIR,
    "track_roi.png",
)

DEFAULT_CELLS_PATH = os.path.join(
    DEFAULT_DATA_DIR,
    "track_roi_cells.txt",
)


# ============================================================
# 1. /odom XY Recorder - ROS2
# ============================================================

class OdomLineRecorder:
    def __init__(
        self,
        line_name: str,
        output_path: str,
        topic: str,
        min_distance: float,
    ):
        """
        ROS2 /odom recorder.

        build / visualize 모드는 ROS와 무관하게 사용할 수 있도록
        ROS2 관련 모듈은 record 모드를 실제 사용할 때만 import한다.
        """

        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from nav_msgs.msg import Odometry

        except ImportError as exc:
            raise RuntimeError(
                "record 모드는 ROS2 Python 환경(rclpy, nav_msgs)이 필요합니다. "
                "먼저 ROS2 setup.bash와 workspace install/setup.bash를 source 하십시오."
            ) from exc

        self.rclpy = rclpy
        self.Node = Node
        self.qos_profile_sensor_data = qos_profile_sensor_data
        self.Odometry = Odometry

        self.line_name = line_name
        self.output_path = output_path
        self.topic = topic
        self.min_distance = float(min_distance)

        self.last_xy = None
        self.count = 0
        self.closed_file = False
        self.node = None

        os.makedirs(
            os.path.dirname(
                os.path.abspath(
                    output_path
                )
            ),
            exist_ok=True,
        )

        self.fp = open(
            output_path,
            "w",
            newline="",
            buffering=1,
            encoding="utf-8",
        )

        self.writer = csv.writer(
            self.fp
        )

        self.writer.writerow(
            [
                "x",
                "y",
                "z",
                "stamp",
                "frame_id",
            ]
        )

    def callback(
        self,
        msg,
    ):
        p = msg.pose.pose.position

        xy = np.array(
            [
                float(p.x),
                float(p.y),
            ],
            dtype=float,
        )

        if self.last_xy is not None:
            distance = float(
                np.linalg.norm(
                    xy
                    -
                    self.last_xy
                )
            )

            if (
                distance
                <
                self.min_distance
            ):
                return

        # ROS2 builtin_interfaces/msg/Time
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

        frame_id = str(
            msg.header.frame_id
        )

        self.writer.writerow(
            [
                float(p.x),
                float(p.y),
                float(p.z),
                stamp,
                frame_id,
            ]
        )

        self.last_xy = xy
        self.count += 1

        if (
            self.count
            %
            100
            ==
            0
        ):
            if self.node is not None:
                self.node.get_logger().info(
                    f"[{self.line_name}] "
                    f"{self.count} points recorded -> "
                    f"{self.output_path}"
                )

    def close(
        self,
    ):
        if self.closed_file:
            return

        self.closed_file = True

        try:
            self.fp.flush()

        finally:
            self.fp.close()

    def run(
        self,
    ):
        node_name = (
            f"track_"
            f"{self.line_name}"
            f"_line_recorder"
        )

        recorder = self

        class RecorderNode(
            self.Node
        ):
            def __init__(
                node_self,
            ):
                super().__init__(
                    node_name
                )

                node_self.subscription = (
                    node_self.create_subscription(
                        recorder.Odometry,
                        recorder.topic,
                        recorder.callback,
                        recorder.qos_profile_sensor_data,
                    )
                )

        self.rclpy.init(
            args=None
        )

        self.node = RecorderNode()

        logger = (
            self.node.get_logger()
        )

        logger.info(
            "============================================"
        )

        logger.info(
            f"Recording line : "
            f"{self.line_name.upper()}"
        )

        logger.info(
            f"Topic          : "
            f"{self.topic}"
        )

        logger.info(
            f"Output         : "
            f"{self.output_path}"
        )

        logger.info(
            f"Min distance   : "
            f"{self.min_distance:.3f} m"
        )

        logger.info(
            "QoS            : SENSOR_DATA / BEST_EFFORT"
        )

        logger.info(
            "Ctrl+C to stop and save"
        )

        logger.info(
            "============================================"
        )

        try:
            self.rclpy.spin(
                self.node
            )

        except KeyboardInterrupt:
            pass

        finally:
            self.close()

            if self.node is not None:
                self.node.destroy_node()
                self.node = None

            if self.rclpy.ok():
                self.rclpy.shutdown()

            print(
                f"\n[{self.line_name}] 저장 완료: "
                f"{self.count} points -> "
                f"{self.output_path}"
            )


# ============================================================
# 2. Geometry Utilities
# ============================================================

def cross2(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    return float(
        a[0] * b[1]
        -
        a[1] * b[0]
    )


def remove_consecutive_duplicates(
    points: np.ndarray,
    tol: float = 1.0e-8,
) -> np.ndarray:
    points = np.asarray(
        points,
        dtype=float,
    )

    if len(points) < 2:
        return points.copy()

    kept = [
        points[0]
    ]

    for p in points[1:]:
        if np.linalg.norm(
            p - kept[-1]
        ) > tol:
            kept.append(p)

    result = np.asarray(
        kept,
        dtype=float,
    )

    if len(result) < 2:
        raise ValueError(
            "서로 다른 라인 좌표가 2개 미만입니다."
        )

    return result


def prepare_polyline(
    points: np.ndarray,
    closed: bool,
) -> np.ndarray:
    pts = remove_consecutive_duplicates(
        points
    )

    if closed:
        if np.linalg.norm(
            pts[-1] - pts[0]
        ) <= 1.0e-8:
            pts[-1] = pts[0]
        else:
            pts = np.vstack(
                [pts, pts[0]]
            )

    return pts


def cumulative_lengths(
    points: np.ndarray,
) -> np.ndarray:
    segment_lengths = np.linalg.norm(
        np.diff(
            points,
            axis=0,
        ),
        axis=1,
    )

    return np.concatenate(
        [
            [0.0],
            np.cumsum(
                segment_lengths
            ),
        ]
    )


def point_at_s(
    points: np.ndarray,
    s: float,
    closed: bool,
) -> np.ndarray:
    cum = cumulative_lengths(
        points
    )

    total = float(
        cum[-1]
    )

    if total <= EPS:
        raise ValueError(
            "Polyline 전체 길이가 0입니다."
        )

    if closed:
        s = float(s) % total

    else:
        s = min(
            max(
                float(s),
                0.0,
            ),
            total,
        )

    if (
        not closed
        and
        abs(s - total) <= EPS
    ):
        return points[-1].copy()

    idx = int(
        np.searchsorted(
            cum,
            s,
            side="right",
        )
        - 1
    )

    idx = max(
        0,
        min(
            idx,
            len(points) - 2,
        ),
    )

    segment_length = (
        cum[idx + 1]
        -
        cum[idx]
    )

    if segment_length <= EPS:
        return points[idx].copy()

    t = (
        s - cum[idx]
    ) / segment_length

    return (
        points[idx]
        +
        t
        *
        (
            points[idx + 1]
            -
            points[idx]
        )
    )


def tangent_at_s(
    points: np.ndarray,
    s: float,
    closed: bool,
    window: float,
) -> np.ndarray:
    total = float(
        cumulative_lengths(
            points
        )[-1]
    )

    if total <= EPS:
        raise ValueError(
            "Polyline 전체 길이가 0입니다."
        )

    half = max(
        float(window) * 0.5,
        1.0e-3,
    )

    if closed:
        p0 = point_at_s(
            points,
            s - half,
            True,
        )

        p1 = point_at_s(
            points,
            s + half,
            True,
        )

    else:
        s0 = max(
            0.0,
            s - half,
        )

        s1 = min(
            total,
            s + half,
        )

        p0 = point_at_s(
            points,
            s0,
            False,
        )

        p1 = point_at_s(
            points,
            s1,
            False,
        )

    vec = p1 - p0
    norm = float(
        np.linalg.norm(vec)
    )

    if norm <= EPS:
        raise ValueError(
            "접선 방향을 계산할 수 없습니다."
        )

    return vec / norm


def project_point_to_polyline(
    query: np.ndarray,
    points: np.ndarray,
) -> Tuple[
    np.ndarray,
    float,
    int,
    float,
]:
    query = np.asarray(
        query,
        dtype=float,
    )

    cum = cumulative_lengths(
        points
    )

    best_dist2 = float("inf")
    best_point = None
    best_s = 0.0
    best_idx = 0
    best_t = 0.0

    for i in range(
        len(points) - 1
    ):
        a = points[i]
        b = points[i + 1]

        ab = b - a

        denominator = float(
            np.dot(
                ab,
                ab,
            )
        )

        if denominator <= EPS:
            continue

        t = float(
            np.dot(
                query - a,
                ab,
            )
            /
            denominator
        )

        t = min(
            max(
                t,
                0.0,
            ),
            1.0,
        )

        projection = (
            a
            +
            t * ab
        )

        dist2 = float(
            np.dot(
                query - projection,
                query - projection,
            )
        )

        if dist2 < best_dist2:
            best_dist2 = dist2
            best_point = projection

            segment_length = float(
                np.linalg.norm(ab)
            )

            best_s = float(
                cum[i]
                +
                t
                *
                segment_length
            )

            best_idx = i
            best_t = t

    if best_point is None:
        raise ValueError(
            "클릭 위치를 기준 라인에 투영할 수 없습니다."
        )

    return (
        best_point,
        best_s,
        best_idx,
        best_t,
    )


def intersect_infinite_line_with_polyline(
    origin: np.ndarray,
    direction: np.ndarray,
    polyline: np.ndarray,
) -> Tuple[
    np.ndarray,
    float,
    float,
]:
    origin = np.asarray(
        origin,
        dtype=float,
    )

    direction = np.asarray(
        direction,
        dtype=float,
    )

    direction_norm = float(
        np.linalg.norm(direction)
    )

    if direction_norm <= EPS:
        raise ValueError(
            "법선 방향 벡터가 0입니다."
        )

    direction = (
        direction
        /
        direction_norm
    )

    cum = cumulative_lengths(
        polyline
    )

    candidates = []

    for i in range(
        len(polyline) - 1
    ):
        a = polyline[i]
        b = polyline[i + 1]

        segment = b - a

        segment_length = float(
            np.linalg.norm(segment)
        )

        if segment_length <= EPS:
            continue

        denominator = cross2(
            direction,
            segment,
        )

        if abs(
            denominator
        ) <= 1.0e-12:
            continue

        qmp = a - origin

        t_line = (
            cross2(
                qmp,
                segment,
            )
            /
            denominator
        )

        u_segment = (
            cross2(
                qmp,
                direction,
            )
            /
            denominator
        )

        if (
            -1.0e-9
            <= u_segment
            <= 1.0 + 1.0e-9
        ):
            u_segment = min(
                max(
                    u_segment,
                    0.0,
                ),
                1.0,
            )

            point = (
                a
                +
                u_segment
                *
                segment
            )

            s = float(
                cum[i]
                +
                u_segment
                *
                segment_length
            )

            candidates.append(
                (
                    abs(
                        float(t_line)
                    ),
                    point,
                    s,
                )
            )

    if not candidates:
        raise ValueError(
            "기준선의 법선과 대상 라인의 교점을 찾지 못했습니다."
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    distance, point, s = (
        candidates[0]
    )

    return (
        point,
        s,
        distance,
    )


def reverse_polyline(
    points: np.ndarray,
    closed: bool,
) -> np.ndarray:
    if closed:
        core = points[:-1]
        reversed_points = (
            core[::-1].copy()
        )

        return np.vstack(
            [
                reversed_points,
                reversed_points[0],
            ]
        )

    return points[::-1].copy()


def resample_with_mandatory_s(
    points: np.ndarray,
    spacing: float,
    mandatory_s: List[float],
    closed: bool,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    if spacing <= 0.0:
        raise ValueError(
            "spacing은 0보다 커야 합니다."
        )

    cum = cumulative_lengths(
        points
    )

    total = float(
        cum[-1]
    )

    if closed:
        base_s = list(
            np.arange(
                0.0,
                total,
                spacing,
            )
        )

        mandatory = [
            float(s % total)
            for s
            in mandatory_s
        ]

    else:
        base_s = list(
            np.arange(
                0.0,
                total,
                spacing,
            )
        )

        if (
            not base_s
            or
            abs(
                base_s[-1] - total
            ) > 1.0e-8
        ):
            base_s.append(
                total
            )

        mandatory = [
            min(
                max(
                    float(s),
                    0.0,
                ),
                total,
            )
            for s
            in mandatory_s
        ]

    all_s = sorted(
        base_s + mandatory
    )

    unique_s = []

    for s in all_s:
        if (
            not unique_s
            or
            abs(
                s - unique_s[-1]
            ) > 1.0e-7
        ):
            unique_s.append(s)

    s_array = np.asarray(
        unique_s,
        dtype=float,
    )

    points_resampled = np.vstack(
        [
            point_at_s(
                points,
                s,
                closed,
            )
            for s
            in s_array
        ]
    )

    return (
        s_array,
        points_resampled,
    )


def extract_section_from_resampled(
    sample_s: np.ndarray,
    sample_points: np.ndarray,
    start_s: float,
    end_s: float,
    total: float,
    closed: bool,
) -> np.ndarray:
    tolerance = 1.0e-7

    if not closed:
        if (
            end_s
            <
            start_s - tolerance
        ):
            raise ValueError(
                "구간 경계 순서가 라인 진행방향과 반대입니다. "
                "트랙 진행순서대로 클릭했는지 확인하십시오."
            )

        mask = (
            (sample_s >= start_s - tolerance)
            &
            (sample_s <= end_s + tolerance)
        )

        section = (
            sample_points[mask]
        )

        if len(section) < 2:
            raise ValueError(
                "구간 내부 재샘플링 점이 부족합니다."
            )

        return section

    start_s = (
        start_s % total
    )

    end_s = (
        end_s % total
    )

    if (
        end_s
        >
        start_s + tolerance
    ):
        mask = (
            (sample_s >= start_s - tolerance)
            &
            (sample_s <= end_s + tolerance)
        )

        section = (
            sample_points[mask]
        )

        if len(section) < 2:
            raise ValueError(
                "구간 내부 재샘플링 점이 부족합니다."
            )

        return section

    part1 = (
        sample_points[
            sample_s
            >= start_s - tolerance
        ]
    )

    part2 = (
        sample_points[
            sample_s
            <= end_s + tolerance
        ]
    )

    section = np.vstack(
        [
            part1,
            part2,
        ]
    )

    if len(section) < 2:
        raise ValueError(
            "폐곡선 구간 재샘플링 점이 부족합니다."
        )

    return section


def close_polygon(
    vertices: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(
        vertices,
        dtype=float,
    )

    if len(vertices) < 3:
        raise ValueError(
            "Polygon은 최소 3개의 꼭짓점이 필요합니다."
        )

    if np.linalg.norm(
        vertices[0] - vertices[-1]
    ) > 1.0e-8:
        vertices = np.vstack(
            [
                vertices,
                vertices[0],
            ]
        )

    return vertices


def polygon_centroid(
    vertices: np.ndarray,
) -> np.ndarray:
    points = np.asarray(
        vertices,
        dtype=float,
    )

    if np.linalg.norm(
        points[0] - points[-1]
    ) <= 1.0e-8:
        points = points[:-1]

    return np.mean(
        points,
        axis=0,
    )


# ============================================================
# 3. File I/O
# ============================================================

def load_line_csv(
    path: str,
) -> Tuple[
    np.ndarray,
    str,
]:
    if not os.path.exists(
        path
    ):
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {path}"
        )

    points = []
    frame_ids = []

    with open(
        path,
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as fp:
        reader = csv.DictReader(
            fp
        )

        required = {
            "x",
            "y",
        }

        if not required.issubset(
            set(
                reader.fieldnames
                or
                []
            )
        ):
            raise ValueError(
                f"{path}: CSV header에 x, y가 필요합니다."
            )

        for row in reader:
            points.append(
                [
                    float(
                        row["x"]
                    ),
                    float(
                        row["y"]
                    ),
                ]
            )

            if (
                "frame_id" in row
                and
                row["frame_id"]
            ):
                frame_ids.append(
                    row["frame_id"]
                )

    if len(points) < 2:
        raise ValueError(
            f"{path}: 유효한 좌표가 2개 미만입니다."
        )

    frame_id = (
        frame_ids[0]
        if frame_ids
        else ""
    )

    return (
        np.asarray(
            points,
            dtype=float,
        ),
        frame_id,
    )


def save_json(
    data: Dict,
    path: str,
):
    os.makedirs(
        os.path.dirname(
            os.path.abspath(path)
        ),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(
            data,
            fp,
            ensure_ascii=False,
            indent=2,
        )


def points_to_list(
    points: np.ndarray,
) -> List[
    List[float]
]:
    return [
        [
            float(p[0]),
            float(p[1]),
        ]
        for p
        in np.asarray(points)
    ]




def polyline_length(
    points: np.ndarray,
) -> float:
    """
    Open polyline 길이 계산.
    """
    pts = np.asarray(
        points,
        dtype=float,
    )

    if len(pts) < 2:
        return 0.0

    return float(
        np.sum(
            np.linalg.norm(
                np.diff(
                    pts,
                    axis=0,
                ),
                axis=1,
            )
        )
    )


def sample_open_polyline_at_s(
    points: np.ndarray,
    s: float,
) -> np.ndarray:
    """
    Open polyline의 누적거리 s 위치를 선형보간한다.
    """
    pts = np.asarray(
        points,
        dtype=float,
    )

    if len(pts) == 0:
        raise ValueError(
            "빈 polyline입니다."
        )

    if len(pts) == 1:
        return pts[0].copy()

    cum = cumulative_lengths(
        pts
    )

    total = float(
        cum[-1]
    )

    if total <= EPS:
        return pts[0].copy()

    s = min(
        max(
            float(s),
            0.0,
        ),
        total,
    )

    if abs(
        s - total
    ) <= EPS:
        return pts[-1].copy()

    index = int(
        np.searchsorted(
            cum,
            s,
            side="right",
        )
        -
        1
    )

    index = max(
        0,
        min(
            index,
            len(pts) - 2,
        ),
    )

    segment_length = float(
        cum[index + 1]
        -
        cum[index]
    )

    if segment_length <= EPS:
        return pts[index].copy()

    ratio = (
        s
        -
        float(
            cum[index]
        )
    ) / segment_length

    return (
        pts[index]
        +
        ratio
        *
        (
            pts[index + 1]
            -
            pts[index]
        )
    )


def resample_boundary_pair_for_cells(
    boundary_a: np.ndarray,
    boundary_b: np.ndarray,
    spacing: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    한 ROI section의 양쪽 경계선을 동일한 개수로 재샘플링한다.

    두 경계선의 전체 길이는 서로 다를 수 있으므로
    각 경계의 동일 진행률(0~1)을 서로 대응시킨다.
    """
    a = remove_consecutive_duplicates(
        np.asarray(
            boundary_a,
            dtype=float,
        )
    )

    b = remove_consecutive_duplicates(
        np.asarray(
            boundary_b,
            dtype=float,
        )
    )

    if len(a) < 2 or len(b) < 2:
        raise ValueError(
            "Cell 생성용 section 경계점이 2개 미만입니다."
        )

    length_a = polyline_length(
        a
    )

    length_b = polyline_length(
        b
    )

    max_length = max(
        length_a,
        length_b,
    )

    if max_length <= EPS:
        raise ValueError(
            "Cell 생성용 section 길이가 0입니다."
        )

    interval_count = max(
        1,
        int(
            np.ceil(
                max_length
                /
                float(spacing)
            )
        ),
    )

    ratios = np.linspace(
        0.0,
        1.0,
        interval_count + 1,
    )

    sampled_a = []
    sampled_b = []

    for ratio in ratios:
        sampled_a.append(
            sample_open_polyline_at_s(
                a,
                ratio * length_a,
            )
        )

        sampled_b.append(
            sample_open_polyline_at_s(
                b,
                ratio * length_b,
            )
        )

    return (
        np.asarray(
            sampled_a,
            dtype=float,
        ),
        np.asarray(
            sampled_b,
            dtype=float,
        ),
    )


def extract_section_boundaries_for_cells(
    section: Dict,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    3-Line / 4-Line section 저장 구조를 모두 지원한다.

    3-Line:
        outer_points / inner_points

    4-Line:
        other_points / reference_points
    """
    if (
        "outer_points" in section
        and
        "inner_points" in section
    ):
        return (
            np.asarray(
                section["outer_points"],
                dtype=float,
            ),
            np.asarray(
                section["inner_points"],
                dtype=float,
            ),
        )

    if (
        "other_points" in section
        and
        "reference_points" in section
    ):
        return (
            np.asarray(
                section["other_points"],
                dtype=float,
            ),
            np.asarray(
                section["reference_points"],
                dtype=float,
            ),
        )

    raise ValueError(
        "Cell 생성에 필요한 section 경계선 데이터를 찾지 못했습니다."
    )


def build_cells_from_sections(
    sections: List[Dict],
    spacing: float,
) -> List[
    List[
        List[float]
    ]
]:
    """
    ROI section들을 트랙 진행 순서대로 4점 Cell로 분할한다.

    각 Cell 꼭짓점:
        boundary A[i]
        boundary A[i+1]
        boundary B[i+1]
        boundary B[i]

    Cell 번호는 save_roi_cells_txt()에서
    IN / OUT 각각 0부터 연속 부여한다.
    """
    cells = []

    for section in sections:
        boundary_a, boundary_b = (
            extract_section_boundaries_for_cells(
                section
            )
        )

        sampled_a, sampled_b = (
            resample_boundary_pair_for_cells(
                boundary_a,
                boundary_b,
                spacing,
            )
        )

        for index in range(
            len(sampled_a) - 1
        ):
            vertices = np.asarray(
                [
                    sampled_a[index],
                    sampled_a[index + 1],
                    sampled_b[index + 1],
                    sampled_b[index],
                ],
                dtype=float,
            )

            unique_points = []

            for point in vertices:
                if not any(
                    np.linalg.norm(
                        point - existing
                    )
                    <=
                    1.0e-8
                    for existing
                    in unique_points
                ):
                    unique_points.append(
                        point
                    )

            if len(unique_points) < 3:
                continue

            cells.append(
                points_to_list(
                    vertices
                )
            )

    return cells


def save_roi_cells_txt(
    data: Dict,
    path: str,
    spacing: float,
) -> None:
    """
    LiDAR ROI filtering용 Cell TXT.

    한 파일 안에서 CELLS_IN / CELLS_OUT을 분리한다.

    형식:
        TRACK_ROI_CELLS_V1

        CELLS_IN N
        CELL 0
        x1 y1
        x2 y2
        x3 y3
        x4 y4
        END_CELL
        ...
        END_CELLS_IN

        CELLS_OUT M
        CELL 0
        ...
        END_CELL
        ...
        END_CELLS_OUT

        END_TRACK_ROI_CELLS
    """
    os.makedirs(
        os.path.dirname(
            os.path.abspath(
                path
            )
        ),
        exist_ok=True,
    )

    roi_sections = data.get(
        "roi_sections",
        {},
    )

    in_cells = build_cells_from_sections(
        roi_sections.get(
            "IN",
            [],
        ),
        spacing,
    )

    out_cells = build_cells_from_sections(
        roi_sections.get(
            "OUT",
            [],
        ),
        spacing,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write(
            "TRACK_ROI_CELLS_V1\n\n"
        )

        fp.write(
            f"CELLS_IN {len(in_cells)}\n"
        )

        for cell_index, vertices in enumerate(
            in_cells
        ):
            fp.write(
                f"CELL {cell_index}\n"
            )

            for x, y in vertices:
                fp.write(
                    f"{float(x):.12f} "
                    f"{float(y):.12f}\n"
                )

            fp.write(
                "END_CELL\n"
            )

        fp.write(
            "END_CELLS_IN\n\n"
        )

        fp.write(
            f"CELLS_OUT {len(out_cells)}\n"
        )

        for cell_index, vertices in enumerate(
            out_cells
        ):
            fp.write(
                f"CELL {cell_index}\n"
            )

            for x, y in vertices:
                fp.write(
                    f"{float(x):.12f} "
                    f"{float(y):.12f}\n"
                )

            fp.write(
                "END_CELL\n"
            )

        fp.write(
            "END_CELLS_OUT\n\n"
        )

        fp.write(
            "END_TRACK_ROI_CELLS\n"
        )

    print(
        f"[OK] IN Cells : {len(in_cells)}"
    )

    print(
        f"[OK] OUT Cells: {len(out_cells)}"
    )


def save_roi_txt(
    data: Dict,
    path: str,
) -> None:
    """
    C++ ROI node에서 std::ifstream으로 읽기 쉬운 단순 TXT 형식.

    형식:
        TRACK_ROI_V1
        LINE_COUNT 3 또는 4
        POLYGON_COUNT N

        POLYGON <id> <roi> <section_type> <track_order> <vertex_count>
        x1 y1
        x2 y2
        ...
        END_POLYGON

        ...
        END_TRACK_ROI

    예:
        POLYGON IN1 IN STRAIGHT 1 85
        ...
        END_POLYGON

        POLYGON OUT1 OUT STRAIGHT 1 82
        ...
        END_POLYGON

    close_polygon() 때문에 마지막 점이 첫 점과 같으면
    TXT 저장 시 마지막 중복점은 제거한다.
    """

    os.makedirs(
        os.path.dirname(
            os.path.abspath(path)
        ),
        exist_ok=True,
    )

    polygons = data.get(
        "polygons",
        [],
    )

    metadata = data.get(
        "metadata",
        {},
    )

    line_count = int(
        metadata.get(
            "line_count",
            0,
        )
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as fp:
        fp.write(
            "TRACK_ROI_V1\n"
        )

        fp.write(
            f"LINE_COUNT {line_count}\n"
        )

        fp.write(
            f"POLYGON_COUNT {len(polygons)}\n"
        )

        for polygon in polygons:
            vertices = np.asarray(
                polygon["vertices"],
                dtype=np.float64,
            )

            # Polygon 닫기용 마지막 중복점 제거
            if (
                len(vertices) >= 2
                and
                np.linalg.norm(
                    vertices[0]
                    -
                    vertices[-1]
                )
                <= 1.0e-8
            ):
                vertices = (
                    vertices[:-1]
                )

            polygon_id = str(
                polygon["id"]
            )

            roi = str(
                polygon["roi"]
            )

            section_type = str(
                polygon[
                    "section_type"
                ]
            )

            track_order = int(
                polygon[
                    "track_order"
                ]
            )

            fp.write(
                "\n"
                f"POLYGON "
                f"{polygon_id} "
                f"{roi} "
                f"{section_type} "
                f"{track_order} "
                f"{len(vertices)}\n"
            )

            for x, y in vertices:
                fp.write(
                    f"{float(x):.12f} "
                    f"{float(y):.12f}\n"
                )

            fp.write(
                "END_POLYGON\n"
            )

        fp.write(
            "\nEND_TRACK_ROI\n"
        )


def ask_line_count() -> int:
    """
    build 시작 시 실제 차선 경계 데이터 개수를 묻는다.

    3개:
      LEFT / MIDDLE / RIGHT

    4개:
      LEFT / MIDDLE1 / MIDDLE2 / RIGHT
    """

    while True:
        raw = input(
            "\n차선 경계 데이터가 몇 개인가요? [3/4]: "
        ).strip()

        if raw == "3":
            return 3

        if raw == "4":
            return 4

        print(
            "3 또는 4를 입력하십시오."
        )


# ============================================================
# 4. Visualization Helpers
# ============================================================

def line_color(
    line_name: str,
) -> str:
    if line_name in (
        "middle",
        "middle1",
        "middle2",
    ):
        return "darkorange"

    return "black"


def line_label(
    line_name: str,
) -> str:
    labels = {
        "left": "LEFT white line",
        "middle": "MIDDLE orange line",
        "middle1": "MIDDLE1 orange line",
        "middle2": "MIDDLE2 orange line",
        "right": "RIGHT white line",
    }

    return labels.get(
        line_name,
        line_name.upper(),
    )


def draw_track_lines(
    ax,
    lines: Dict[
        str,
        np.ndarray,
    ],
    scatter: bool = False,
):
    for name, points in lines.items():
        points = np.asarray(
            points,
            dtype=float,
        )

        color = line_color(
            name
        )

        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=(
                2.0
                if "middle" in name
                else 1.8
            ),
            label=line_label(
                name
            ),
        )

        if scatter:
            ax.scatter(
                points[:, 0],
                points[:, 1],
                s=7,
                color=color,
            )


def draw_polygon_list(
    ax,
    polygons: List[Dict],
    show_label: bool = True,
):
    for polygon in polygons:
        vertices = np.asarray(
            polygon["vertices"],
            dtype=float,
        )

        roi = str(
            polygon.get(
                "roi",
                "",
            )
        ).upper()

        section_type = str(
            polygon.get(
                "section_type",
                "",
            )
        ).upper()

        color_map = {
            ("IN", "STRAIGHT"): "green",
            ("IN", "CURVE"): "red",
            ("OUT", "STRAIGHT"): "blue",
            ("OUT", "CURVE"): "yellow",
        }

        color = color_map.get(
            (
                roi,
                section_type,
            ),
            "gray",
        )

        patch = MplPolygon(
            vertices,
            closed=True,
            facecolor=color,
            edgecolor=color,
            alpha=0.20,
            linewidth=1.4,
        )

        ax.add_patch(
            patch
        )

        if show_label:
            center = polygon_centroid(
                vertices
            )

            ax.text(
                center[0],
                center[1],
                (
                    f"{polygon['id']}\n"
                    f"{polygon['section_name'].upper()}"
                ),
                ha="center",
                va="center",
                fontsize=8,
            )


# ============================================================
# 5. Section Type
# ============================================================

def make_section_types(
    section_count: int,
) -> List[str]:
    """
    첫 번째 클릭점 -> 두 번째 클릭점은 무조건 STRAIGHT.

    이후:
      CURVE -> STRAIGHT -> CURVE -> STRAIGHT ...
    """

    result = []

    for i in range(
        section_count
    ):
        if i % 2 == 0:
            result.append(
                "STRAIGHT"
            )

        else:
            result.append(
                "CURVE"
            )

    return result


# ============================================================
# 6. Three-Line Boundary Model
# ============================================================

@dataclass
class ThreeLineBoundary:
    middle_point: np.ndarray
    middle_s: float
    tangent: np.ndarray
    normal: np.ndarray

    left_point: Optional[np.ndarray] = None
    left_s: Optional[float] = None

    right_point: Optional[np.ndarray] = None
    right_s: Optional[float] = None

    kind: str = "transition"


def make_three_boundary_at_s(
    middle: np.ndarray,
    s: float,
    closed: bool,
    tangent_window: float,
    kind: str,
) -> ThreeLineBoundary:
    point = point_at_s(
        middle,
        s,
        closed,
    )

    tangent = tangent_at_s(
        middle,
        s,
        closed,
        tangent_window,
    )

    normal = np.array(
        [
            -tangent[1],
            tangent[0],
        ],
        dtype=float,
    )

    return ThreeLineBoundary(
        middle_point=point,
        middle_s=float(s),
        tangent=tangent,
        normal=normal,
        kind=kind,
    )


def select_three_line_boundaries(
    left: np.ndarray,
    middle: np.ndarray,
    right: np.ndarray,
    closed: bool,
    tangent_window: float,
) -> List[
    ThreeLineBoundary
]:
    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    draw_track_lines(
        ax,
        {
            "left": left,
            "middle": middle,
            "right": right,
        },
        scatter=False,
    )

    ax.scatter(
        middle[:, 0],
        middle[:, 1],
        s=7,
        color="darkorange",
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True
    )

    ax.legend()

    ax.set_title(
        "Click transition points on MIDDLE line in track order\n"
        "Left click: Add | Right click: Undo last | "
        "Enter/Middle click: Finish"
    )

    ax.set_xlabel(
        "x [m]"
    )

    ax.set_ylabel(
        "y [m]"
    )

    print(
        "\n============================================"
    )
    print(
        "3-Line 모드: 공통 구간 경계 선택"
    )
    print(
        "============================================"
    )
    print(
        "MIDDLE line 위에서 트랙 진행순서대로 클릭"
    )
    print(
        "P1 -> P2 = STRAIGHT1"
    )
    print(
        "P2 -> P3 = CURVE1"
    )
    print(
        "이후 STRAIGHT / CURVE 반복"
    )
    print(
        "Enter 또는 가운데 클릭으로 완료"
    )
    print(
        "============================================"
    )

    clicked = plt.ginput(
        n=-1,
        timeout=0,
        show_clicks=True,
    )

    plt.close(
        fig
    )

    if (
        closed
        and
        len(clicked) < 2
    ):
        raise ValueError(
            "폐곡선 트랙은 최소 2개의 경계점이 필요합니다."
        )

    result = []

    for x, y in clicked:
        projection, s, _, _ = (
            project_point_to_polyline(
                np.array(
                    [x, y],
                    dtype=float,
                ),
                middle,
            )
        )

        tangent = tangent_at_s(
            middle,
            s,
            closed,
            tangent_window,
        )

        normal = np.array(
            [
                -tangent[1],
                tangent[0],
            ],
            dtype=float,
        )

        result.append(
            ThreeLineBoundary(
                middle_point=projection,
                middle_s=float(s),
                tangent=tangent,
                normal=normal,
                kind="transition",
            )
        )

    return remove_duplicate_three_boundaries(
        result,
        closed,
    )


def remove_duplicate_three_boundaries(
    boundaries: List[
        ThreeLineBoundary
    ],
    closed: bool,
) -> List[
    ThreeLineBoundary
]:
    filtered = []

    for boundary in boundaries:
        duplicate = any(
            abs(
                boundary.middle_s
                -
                previous.middle_s
            )
            <
            1.0e-4

            for previous
            in filtered
        )

        if not duplicate:
            filtered.append(
                boundary
            )

    if (
        len(filtered)
        !=
        len(boundaries)
    ):
        print(
            "[WARN] 중복으로 판단된 경계점은 제거했습니다."
        )

    if (
        closed
        and
        len(filtered) < 2
    ):
        raise ValueError(
            "중복 제거 후 유효 경계점이 2개 미만입니다."
        )

    return filtered


def populate_three_line_intersections(
    boundaries: List[
        ThreeLineBoundary
    ],
    left: np.ndarray,
    right: np.ndarray,
):
    for boundary in boundaries:
        left_point, left_s, _ = (
            intersect_infinite_line_with_polyline(
                boundary.middle_point,
                boundary.normal,
                left,
            )
        )

        right_point, right_s, _ = (
            intersect_infinite_line_with_polyline(
                boundary.middle_point,
                boundary.normal,
                right,
            )
        )

        boundary.left_point = (
            left_point
        )

        boundary.left_s = float(
            left_s
        )

        boundary.right_point = (
            right_point
        )

        boundary.right_s = float(
            right_s
        )


def three_side_orientation_score(
    boundaries: List[
        ThreeLineBoundary
    ],
    side_points: np.ndarray,
    side_key: str,
    closed: bool,
    tangent_window: float,
) -> float:
    values = []

    for boundary in boundaries:
        if side_key == "left":
            side_s = boundary.left_s

        else:
            side_s = boundary.right_s

        side_tangent = tangent_at_s(
            side_points,
            side_s,
            closed,
            tangent_window,
        )

        values.append(
            float(
                np.dot(
                    side_tangent,
                    boundary.tangent,
                )
            )
        )

    if not values:
        return 1.0

    return float(
        np.mean(values)
    )


# ============================================================
# 7. Four-Line Pair Boundary Model
# ============================================================

@dataclass
class PairBoundary:
    reference_point: np.ndarray
    reference_s: float
    tangent: np.ndarray
    normal: np.ndarray

    other_point: Optional[np.ndarray] = None
    other_s: Optional[float] = None

    kind: str = "transition"


def make_pair_boundary_at_s(
    reference: np.ndarray,
    s: float,
    closed: bool,
    tangent_window: float,
    kind: str,
) -> PairBoundary:
    point = point_at_s(
        reference,
        s,
        closed,
    )

    tangent = tangent_at_s(
        reference,
        s,
        closed,
        tangent_window,
    )

    normal = np.array(
        [
            -tangent[1],
            tangent[0],
        ],
        dtype=float,
    )

    return PairBoundary(
        reference_point=point,
        reference_s=float(s),
        tangent=tangent,
        normal=normal,
        kind=kind,
    )


def select_pair_boundaries(
    roi_name: str,
    reference_name: str,
    reference: np.ndarray,
    other: np.ndarray,
    all_lines: Dict[
        str,
        np.ndarray,
    ],
    closed: bool,
    tangent_window: float,
    background_polygons: Optional[
        List[Dict]
    ] = None,
) -> List[
    PairBoundary
]:
    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    if background_polygons:
        draw_polygon_list(
            ax,
            background_polygons,
            show_label=True,
        )

    draw_track_lines(
        ax,
        all_lines,
        scatter=False,
    )

    ax.scatter(
        reference[:, 0],
        reference[:, 1],
        s=8,
        color=line_color(
            reference_name
        ),
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True
    )

    ax.legend()

    if (
        roi_name
        ==
        "IN"
    ):
        title = (
            "Select IN section boundaries on MIDDLE1 for one lap\n"
            "Left click: Add | Right click: Undo last | "
            "Enter/Middle click: Generate IN"
        )

    else:
        title = (
            "Check IN polygons, then select OUT section boundaries\n"
            "Click one lap on MIDDLE2 | "
            "Enter/Middle click: Generate OUT"
        )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "x [m]"
    )

    ax.set_ylabel(
        "y [m]"
    )

    print(
        "\n============================================"
    )
    print(
        f"4-Line 모드: {roi_name} 경계 선택"
    )
    print(
        "============================================"
    )
    print(
        f"기준선: {reference_name.upper()}"
    )
    print(
        "트랙 진행순서대로 한 바퀴 클릭"
    )
    print(
        "첫 번째 구간은 STRAIGHT"
    )
    print(
        "이후 CURVE / STRAIGHT 자동 반복"
    )
    print(
        "Enter 또는 가운데 클릭으로 완료"
    )
    print(
        "============================================"
    )

    clicked = plt.ginput(
        n=-1,
        timeout=0,
        show_clicks=True,
    )

    plt.close(
        fig
    )

    if (
        closed
        and
        len(clicked) < 2
    ):
        raise ValueError(
            f"{roi_name} 폐곡선 구간은 최소 2개의 경계점이 필요합니다."
        )

    result = []

    for x, y in clicked:
        projection, s, _, _ = (
            project_point_to_polyline(
                np.array(
                    [x, y],
                    dtype=float,
                ),
                reference,
            )
        )

        tangent = tangent_at_s(
            reference,
            s,
            closed,
            tangent_window,
        )

        normal = np.array(
            [
                -tangent[1],
                tangent[0],
            ],
            dtype=float,
        )

        result.append(
            PairBoundary(
                reference_point=projection,
                reference_s=float(s),
                tangent=tangent,
                normal=normal,
                kind="transition",
            )
        )

    return remove_duplicate_pair_boundaries(
        result,
        closed,
        roi_name,
    )


def remove_duplicate_pair_boundaries(
    boundaries: List[
        PairBoundary
    ],
    closed: bool,
    roi_name: str,
) -> List[
    PairBoundary
]:
    filtered = []

    for boundary in boundaries:
        duplicate = any(
            abs(
                boundary.reference_s
                -
                previous.reference_s
            )
            <
            1.0e-4

            for previous
            in filtered
        )

        if not duplicate:
            filtered.append(
                boundary
            )

    if (
        len(filtered)
        !=
        len(boundaries)
    ):
        print(
            f"[WARN] {roi_name}: 중복으로 판단된 경계점은 제거했습니다."
        )

    if (
        closed
        and
        len(filtered) < 2
    ):
        raise ValueError(
            f"{roi_name}: 중복 제거 후 유효 경계점이 2개 미만입니다."
        )

    return filtered


def populate_pair_intersections(
    boundaries: List[
        PairBoundary
    ],
    other: np.ndarray,
):
    for boundary in boundaries:
        point, s, _ = (
            intersect_infinite_line_with_polyline(
                boundary.reference_point,
                boundary.normal,
                other,
            )
        )

        boundary.other_point = (
            point
        )

        boundary.other_s = float(
            s
        )


def pair_orientation_score(
    boundaries: List[
        PairBoundary
    ],
    other: np.ndarray,
    closed: bool,
    tangent_window: float,
) -> float:
    values = []

    for boundary in boundaries:
        other_tangent = tangent_at_s(
            other,
            boundary.other_s,
            closed,
            tangent_window,
        )

        values.append(
            float(
                np.dot(
                    other_tangent,
                    boundary.tangent,
                )
            )
        )

    if not values:
        return 1.0

    return float(
        np.mean(values)
    )


# ============================================================
# 8. Common Section Pair
# ============================================================

def make_section_pairs(
    boundaries,
    closed: bool,
):
    if closed:
        return [
            (
                boundaries[i],
                boundaries[
                    (i + 1)
                    %
                    len(boundaries)
                ],
            )
            for i
            in range(
                len(boundaries)
            )
        ]

    return [
        (
            boundaries[i],
            boundaries[i + 1],
        )
        for i
        in range(
            len(boundaries) - 1
        )
    ]


# ============================================================
# 9. Three-Line ROI Build
# ============================================================

def assemble_three_line_roi(
    args,
    left: np.ndarray,
    middle: np.ndarray,
    right: np.ndarray,
    boundaries: List[ThreeLineBoundary],
    frame_id: str,
) -> Dict:
    """
    현재 ThreeLineBoundary 목록을 기준으로
    IN / OUT section, polygon, boundary JSON 데이터를 다시 만든다.

    Interactive editor에서 경계를 이동한 뒤에도
    이 함수를 다시 호출하여 동일한 형식의 최종 결과를 생성한다.
    """

    middle_total = float(
        cumulative_lengths(
            middle
        )[-1]
    )

    left_total = float(
        cumulative_lengths(
            left
        )[-1]
    )

    right_total = float(
        cumulative_lengths(
            right
        )[-1]
    )

    section_pairs = make_section_pairs(
        boundaries,
        args.closed,
    )

    section_types = make_section_types(
        len(section_pairs)
    )

    left_s, left_resampled = (
        resample_with_mandatory_s(
            left,
            args.spacing,
            [
                boundary.left_s
                for boundary
                in boundaries
            ],
            args.closed,
        )
    )

    middle_s, middle_resampled = (
        resample_with_mandatory_s(
            middle,
            args.spacing,
            [
                boundary.middle_s
                for boundary
                in boundaries
            ],
            args.closed,
        )
    )

    right_s, right_resampled = (
        resample_with_mandatory_s(
            right,
            args.spacing,
            [
                boundary.right_s
                for boundary
                in boundaries
            ],
            args.closed,
        )
    )

    polygons = []
    in_sections = []
    out_sections = []

    straight_number = 0
    curve_number = 0

    for order_index, (
        boundary_pair,
        section_type,
    ) in enumerate(
        zip(
            section_pairs,
            section_types,
        ),
        start=1,
    ):
        start_boundary, end_boundary = (
            boundary_pair
        )

        if section_type == "STRAIGHT":
            straight_number += 1

            section_type_index = (
                straight_number
            )

            section_name = (
                f"straight"
                f"{section_type_index}"
            )

        else:
            curve_number += 1

            section_type_index = (
                curve_number
            )

            section_name = (
                f"curve"
                f"{section_type_index}"
            )

        left_section = (
            extract_section_from_resampled(
                left_s,
                left_resampled,
                start_boundary.left_s,
                end_boundary.left_s,
                left_total,
                args.closed,
            )
        )

        middle_section = (
            extract_section_from_resampled(
                middle_s,
                middle_resampled,
                start_boundary.middle_s,
                end_boundary.middle_s,
                middle_total,
                args.closed,
            )
        )

        right_section = (
            extract_section_from_resampled(
                right_s,
                right_resampled,
                start_boundary.right_s,
                end_boundary.right_s,
                right_total,
                args.closed,
            )
        )

        in_vertices = close_polygon(
            np.vstack(
                [
                    left_section,
                    middle_section[::-1],
                ]
            )
        )

        out_vertices = close_polygon(
            np.vstack(
                [
                    middle_section,
                    right_section[::-1],
                ]
            )
        )

        in_id = (
            f"IN{order_index}"
        )

        out_id = (
            f"OUT{order_index}"
        )

        in_section_data = {
            "roi": "IN",
            "track_order": order_index,
            "section_type": section_type,
            "section_type_index": section_type_index,
            "section_name": section_name,
            "line_names": {
                "outer": f"left_line_{section_name}",
                "inner": f"middle_line_{section_name}",
            },
            "outer_points": points_to_list(
                left_section
            ),
            "inner_points": points_to_list(
                middle_section
            ),
            "polygon_id": in_id,
        }

        out_section_data = {
            "roi": "OUT",
            "track_order": order_index,
            "section_type": section_type,
            "section_type_index": section_type_index,
            "section_name": section_name,
            "line_names": {
                "inner": f"middle_line_{section_name}",
                "outer": f"right_line_{section_name}",
            },
            "inner_points": points_to_list(
                middle_section
            ),
            "outer_points": points_to_list(
                right_section
            ),
            "polygon_id": out_id,
        }

        in_sections.append(
            in_section_data
        )

        out_sections.append(
            out_section_data
        )

        polygons.append(
            {
                "id": in_id,
                "roi": "IN",
                "track_order": order_index,
                "section_type": section_type,
                "section_type_index": section_type_index,
                "section_name": section_name,
                "vertices": points_to_list(
                    in_vertices
                ),
            }
        )

        polygons.append(
            {
                "id": out_id,
                "roi": "OUT",
                "track_order": order_index,
                "section_type": section_type,
                "section_type_index": section_type_index,
                "section_name": section_name,
                "vertices": points_to_list(
                    out_vertices
                ),
            }
        )

    boundary_data = []

    for index, boundary in enumerate(
        boundaries,
        start=1,
    ):
        boundary_data.append(
            {
                "roi": "SHARED",
                "id": index,
                "kind": boundary.kind,
                "points": [
                    points_to_list(
                        [boundary.left_point]
                    )[0],
                    points_to_list(
                        [boundary.middle_point]
                    )[0],
                    points_to_list(
                        [boundary.right_point]
                    )[0],
                ],
                "left_s": float(
                    boundary.left_s
                ),
                "middle_s": float(
                    boundary.middle_s
                ),
                "right_s": float(
                    boundary.right_s
                ),
            }
        )

    return {
        "metadata": {
            "coordinate_source": "/odom pose.position.x/y",
            "line_count": 3,
            "line_mode": "LEFT / MIDDLE / RIGHT",
            "frame_id": frame_id,
            "closed_track": bool(
                args.closed
            ),
            "resample_spacing_m": float(
                args.spacing
            ),
            "tangent_window_m": float(
                args.tangent_window
            ),
            "roi_definition": {
                "IN": "LEFT <-> MIDDLE",
                "OUT": "MIDDLE <-> RIGHT",
            },
        },
        "resampled_lines": {
            "left": points_to_list(
                left_resampled
            ),
            "middle": points_to_list(
                middle_resampled
            ),
            "right": points_to_list(
                right_resampled
            ),
        },
        "boundaries": boundary_data,
        "roi_sections": {
            "IN": in_sections,
            "OUT": out_sections,
        },
        "polygons": polygons,
    }


def forward_s_distance(
    start_s: float,
    end_s: float,
    total_s: float,
) -> float:
    """
    Closed path에서 start_s -> end_s의
    진행 방향 누적거리.
    """

    return (
        float(end_s)
        -
        float(start_s)
    ) % float(total_s)


def three_boundary_candidate_is_valid(
    boundaries: List[ThreeLineBoundary],
    index: int,
    candidate_s: float,
    middle_total: float,
    closed: bool,
    margin: float = 1.0e-3,
) -> bool:
    """
    드래그한 경계가 인접 경계를 넘어가지 않도록 검사한다.

    이유:
      경계 순서를 바꾸면
      STRAIGHT / CURVE의 track_order 의미가 달라질 수 있다.
    """

    count = len(
        boundaries
    )

    if count <= 1:
        return True

    if not closed:
        if boundaries[index].kind != "transition":
            return False

        lower = 0.0
        upper = middle_total

        if index > 0:
            lower = (
                boundaries[index - 1].middle_s
                +
                margin
            )

        if index < count - 1:
            upper = (
                boundaries[index + 1].middle_s
                -
                margin
            )

        return (
            lower
            <
            candidate_s
            <
            upper
        )

    # 경계가 2개뿐이면 서로 같은 위치로 겹치는 것만 방지한다.
    if count == 2:
        other_index = (
            1 - index
        )

        distance_to_other = min(
            forward_s_distance(
                candidate_s,
                boundaries[
                    other_index
                ].middle_s,
                middle_total,
            ),
            forward_s_distance(
                boundaries[
                    other_index
                ].middle_s,
                candidate_s,
                middle_total,
            ),
        )

        return (
            distance_to_other
            >
            margin
        )

    previous_index = (
        index - 1
    ) % count

    next_index = (
        index + 1
    ) % count

    previous_s = (
        boundaries[
            previous_index
        ].middle_s
    )

    next_s = (
        boundaries[
            next_index
        ].middle_s
    )

    whole_interval = (
        forward_s_distance(
            previous_s,
            next_s,
            middle_total,
        )
    )

    candidate_interval = (
        forward_s_distance(
            previous_s,
            candidate_s,
            middle_total,
        )
    )

    return (
        candidate_interval
        >
        margin
        and
        candidate_interval
        <
        whole_interval
        -
        margin
    )


def replace_three_boundary_at_s(
    boundaries: List[ThreeLineBoundary],
    index: int,
    new_s: float,
    middle: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    args,
):
    """
    한 개 공유 경계를 MIDDLE상의 new_s로 이동시키고
    LEFT / RIGHT 교차점까지 다시 계산한다.
    """

    old_kind = (
        boundaries[index].kind
    )

    new_boundary = (
        make_three_boundary_at_s(
            middle,
            new_s,
            args.closed,
            args.tangent_window,
            old_kind,
        )
    )

    left_point, left_s, _ = (
        intersect_infinite_line_with_polyline(
            new_boundary.middle_point,
            new_boundary.normal,
            left,
        )
    )

    right_point, right_s, _ = (
        intersect_infinite_line_with_polyline(
            new_boundary.middle_point,
            new_boundary.normal,
            right,
        )
    )

    new_boundary.left_point = (
        left_point
    )

    new_boundary.left_s = float(
        left_s
    )

    new_boundary.right_point = (
        right_point
    )

    new_boundary.right_s = float(
        right_s
    )

    boundaries[index] = (
        new_boundary
    )


def restore_three_boundaries_from_s(
    boundaries: List[ThreeLineBoundary],
    original_s_values: List[float],
    middle: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    args,
):
    for index, original_s in enumerate(
        original_s_values
    ):
        replace_three_boundary_at_s(
            boundaries,
            index,
            original_s,
            middle,
            left,
            right,
            args,
        )


def edit_three_line_boundaries_interactive(
    args,
    left: np.ndarray,
    middle: np.ndarray,
    right: np.ndarray,
    boundaries: List[ThreeLineBoundary],
    initial_result: Dict,
    frame_id: str,
) -> Dict:
    """
    3-Line ROI 결과 확인/수정 창.

    조작:
      Left drag : 빨간 공유 경계 이동
      Enter     : 현재 수정 상태 확정
      Esc       : 모든 수정 취소

    경계의 가운데 MIDDLE 점을 잡아 드래그한다.
    커서 위치는 MIDDLE polyline 위로 자동 projection된다.
    """

    if args.no_show:
        print(
            "[INFO] --no-show 사용 중이므로 "
            "interactive boundary editor를 건너뜁니다."
        )

        return initial_result

    middle_total = float(
        cumulative_lengths(
            middle
        )[-1]
    )

    original_s_values = [
        float(
            boundary.middle_s
        )
        for boundary
        in boundaries
    ]

    state = {
        "drag_index": None,
        "drag_start_s": None,
        "committed": False,
        "cancelled": False,
        "result": initial_result,
        "boundary_line_artists": [],
        "handle_scatter": None,
    }

    fig, ax = plt.subplots(
        figsize=(14, 10)
    )

    def redraw():
        ax.clear()

        draw_polygon_list(
            ax,
            state["result"]["polygons"],
            show_label=True,
        )

        draw_track_lines(
            ax,
            {
                "left": np.asarray(
                    state["result"][
                        "resampled_lines"
                    ]["left"],
                    dtype=float,
                ),
                "middle": np.asarray(
                    state["result"][
                        "resampled_lines"
                    ]["middle"],
                    dtype=float,
                ),
                "right": np.asarray(
                    state["result"][
                        "resampled_lines"
                    ]["right"],
                    dtype=float,
                ),
            },
            scatter=False,
        )

        state[
            "boundary_line_artists"
        ] = []

        middle_handles = []

        for boundary_index, boundary in enumerate(
            boundaries
        ):
            points = np.asarray(
                [
                    boundary.left_point,
                    boundary.middle_point,
                    boundary.right_point,
                ],
                dtype=float,
            )

            line_artist, = ax.plot(
                points[:, 0],
                points[:, 1],
                "--",
                color="crimson",
                linewidth=1.3,
                alpha=0.95,
                zorder=8,
                label=(
                    "Draggable shared boundary"
                    if boundary_index == 0
                    else None
                ),
            )

            state[
                "boundary_line_artists"
            ].append(
                line_artist
            )

            middle_handles.append(
                boundary.middle_point
            )

            ax.text(
                boundary.middle_point[0],
                boundary.middle_point[1],
                f"B{boundary_index + 1}",
                fontsize=8,
                ha="left",
                va="bottom",
                zorder=10,
            )

        middle_handles_array = np.asarray(
            middle_handles,
            dtype=float,
        )

        state[
            "handle_scatter"
        ] = ax.scatter(
            middle_handles_array[:, 0],
            middle_handles_array[:, 1],
            s=65,
            color="crimson",
            edgecolors="black",
            linewidths=0.6,
            zorder=11,
            label="Drag handle",
        )

        ax.set_aspect(
            "equal",
            adjustable="box",
        )

        ax.grid(
            True
        )

        ax.set_xlabel(
            "x [m]"
        )

        ax.set_ylabel(
            "y [m]"
        )

        ax.set_title(
            "3-Line ROI Boundary Editor\n"
            "Drag a red boundary handle | Enter: Save | Esc: Cancel"
        )

        ax.legend(
            loc="best"
        )

        fig.tight_layout()
        fig.canvas.draw_idle()

    def update_drag_artists(
        boundary_index: int,
    ):
        boundary = (
            boundaries[
                boundary_index
            ]
        )

        points = np.asarray(
            [
                boundary.left_point,
                boundary.middle_point,
                boundary.right_point,
            ],
            dtype=float,
        )

        line_artist = (
            state[
                "boundary_line_artists"
            ][boundary_index]
        )

        line_artist.set_data(
            points[:, 0],
            points[:, 1],
        )

        offsets = np.asarray(
            [
                boundary.middle_point
                for boundary
                in boundaries
            ],
            dtype=float,
        )

        state[
            "handle_scatter"
        ].set_offsets(
            offsets
        )

        fig.canvas.draw_idle()

    def pick_boundary_index(
        event,
    ):
        if (
            event.xdata is None
            or
            event.ydata is None
        ):
            return None

        cursor = np.array(
            [
                event.xdata,
                event.ydata,
            ],
            dtype=float,
        )

        handles = np.asarray(
            [
                boundary.middle_point
                for boundary
                in boundaries
            ],
            dtype=float,
        )

        distances = np.linalg.norm(
            handles
            -
            cursor,
            axis=1,
        )

        nearest_index = int(
            np.argmin(
                distances
            )
        )

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        diagonal = float(
            np.hypot(
                xlim[1] - xlim[0],
                ylim[1] - ylim[0],
            )
        )

        pick_radius = max(
            diagonal * 0.02,
            0.5,
        )

        if (
            distances[
                nearest_index
            ]
            <=
            pick_radius
        ):
            return nearest_index

        return None

    def on_press(
        event,
    ):
        if (
            event.inaxes is not ax
            or
            event.button != 1
        ):
            return

        boundary_index = (
            pick_boundary_index(
                event
            )
        )

        if boundary_index is None:
            return

        if (
            not args.closed
            and
            boundaries[
                boundary_index
            ].kind
            !=
            "transition"
        ):
            print(
                "[WARN] Track start/end boundary는 이동하지 않습니다."
            )

            return

        state[
            "drag_index"
        ] = boundary_index

        state[
            "drag_start_s"
        ] = float(
            boundaries[
                boundary_index
            ].middle_s
        )

        print(
            f"[EDIT] B{boundary_index + 1} drag start"
        )

    def on_motion(
        event,
    ):
        boundary_index = (
            state[
                "drag_index"
            ]
        )

        if boundary_index is None:
            return

        if (
            event.inaxes is not ax
            or
            event.xdata is None
            or
            event.ydata is None
        ):
            return

        try:
            _, candidate_s, _, _ = (
                project_point_to_polyline(
                    np.array(
                        [
                            event.xdata,
                            event.ydata,
                        ],
                        dtype=float,
                    ),
                    middle,
                )
            )

            candidate_s = float(
                candidate_s
            )

            if not three_boundary_candidate_is_valid(
                boundaries,
                boundary_index,
                candidate_s,
                middle_total,
                args.closed,
            ):
                return

            replace_three_boundary_at_s(
                boundaries,
                boundary_index,
                candidate_s,
                middle,
                left,
                right,
                args,
            )

            update_drag_artists(
                boundary_index
            )

        except Exception:
            # 이동 중 교차점이 형성되지 않는 순간은 무시한다.
            return

    def on_release(
        event,
    ):
        boundary_index = (
            state[
                "drag_index"
            ]
        )

        if boundary_index is None:
            return

        try:
            state["result"] = (
                assemble_three_line_roi(
                    args,
                    left,
                    middle,
                    right,
                    boundaries,
                    frame_id,
                )
            )

            print(
                f"[EDIT] B{boundary_index + 1} updated: "
                f"s={boundaries[boundary_index].middle_s:.3f} m"
            )

        except Exception as exc:
            print(
                f"[WARN] 경계 수정 결과를 만들 수 없어 "
                f"B{boundary_index + 1}를 이전 위치로 복구합니다."
            )

            print(
                f"[WARN] {exc}"
            )

            replace_three_boundary_at_s(
                boundaries,
                boundary_index,
                state[
                    "drag_start_s"
                ],
                middle,
                left,
                right,
                args,
            )

            state["result"] = (
                assemble_three_line_roi(
                    args,
                    left,
                    middle,
                    right,
                    boundaries,
                    frame_id,
                )
            )

        state[
            "drag_index"
        ] = None

        state[
            "drag_start_s"
        ] = None

        redraw()

    def on_key(
        event,
    ):
        if event.key == "enter":
            state[
                "committed"
            ] = True

            print(
                "\n[OK] Boundary edit confirmed."
            )

            plt.close(
                fig
            )

        elif event.key == "escape":
            state[
                "cancelled"
            ] = True

            restore_three_boundaries_from_s(
                boundaries,
                original_s_values,
                middle,
                left,
                right,
                args,
            )

            state["result"] = (
                assemble_three_line_roi(
                    args,
                    left,
                    middle,
                    right,
                    boundaries,
                    frame_id,
                )
            )

            print(
                "\n[INFO] Boundary edits cancelled."
            )

            plt.close(
                fig
            )

    fig.canvas.mpl_connect(
        "button_press_event",
        on_press,
    )

    fig.canvas.mpl_connect(
        "motion_notify_event",
        on_motion,
    )

    fig.canvas.mpl_connect(
        "button_release_event",
        on_release,
    )

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key,
    )

    redraw()

    print(
        "\n============================================"
    )
    print(
        "3-Line ROI Boundary Editor"
    )
    print(
        "============================================"
    )
    print(
        "빨간 공유 경계의 MIDDLE 점을 마우스로 드래그"
    )
    print(
        "경계는 MIDDLE line 위로 자동 투영"
    )
    print(
        "인접 경계를 넘어가는 이동은 차단"
    )
    print(
        "마우스를 놓으면 IN/OUT Polygon 즉시 재생성"
    )
    print(
        "Enter : 현재 수정 결과 확정 후 저장 진행"
    )
    print(
        "Esc   : 모든 수정 취소"
    )
    print(
        "============================================"
    )

    plt.show()

    if not state["committed"]:
        if not state["cancelled"]:
            print(
                "[INFO] Enter 없이 창을 닫았습니다. "
                "수정 전 결과를 사용합니다."
            )

            restore_three_boundaries_from_s(
                boundaries,
                original_s_values,
                middle,
                left,
                right,
                args,
            )

            state["result"] = (
                assemble_three_line_roi(
                    args,
                    left,
                    middle,
                    right,
                    boundaries,
                    frame_id,
                )
            )

    return state["result"]


def build_three_line_roi(
    args,
    data_dir: str,
) -> Dict:
    left_path = os.path.join(
        data_dir,
        "left_raw.csv",
    )

    middle_path = os.path.join(
        data_dir,
        "middle_raw.csv",
    )

    right_path = os.path.join(
        data_dir,
        "right_raw.csv",
    )

    left_raw, left_frame = (
        load_line_csv(
            left_path
        )
    )

    middle_raw, middle_frame = (
        load_line_csv(
            middle_path
        )
    )

    right_raw, right_frame = (
        load_line_csv(
            right_path
        )
    )

    frame_ids = [
        frame
        for frame
        in [
            left_frame,
            middle_frame,
            right_frame,
        ]
        if frame
    ]

    if (
        frame_ids
        and
        len(
            set(frame_ids)
        )
        != 1
    ):
        raise ValueError(
            "LEFT/MIDDLE/RIGHT의 frame_id가 서로 다릅니다."
        )

    frame_id = (
        frame_ids[0]
        if frame_ids
        else ""
    )

    left = prepare_polyline(
        left_raw,
        args.closed,
    )

    middle = prepare_polyline(
        middle_raw,
        args.closed,
    )

    right = prepare_polyline(
        right_raw,
        args.closed,
    )

    middle_total = float(
        cumulative_lengths(
            middle
        )[-1]
    )

    # --------------------------------------------------------
    # STEP 1: 기존 방식으로 초기 경계 클릭
    # --------------------------------------------------------

    boundaries = (
        select_three_line_boundaries(
            left,
            middle,
            right,
            args.closed,
            args.tangent_window,
        )
    )

    if not args.closed:
        boundaries = [
            make_three_boundary_at_s(
                middle,
                0.0,
                False,
                args.tangent_window,
                "track_start",
            ),
            *boundaries,
            make_three_boundary_at_s(
                middle,
                middle_total,
                False,
                args.tangent_window,
                "track_end",
            ),
        ]

    populate_three_line_intersections(
        boundaries,
        left,
        right,
    )

    # --------------------------------------------------------
    # LEFT / RIGHT 진행방향 보정
    # --------------------------------------------------------

    left_score = (
        three_side_orientation_score(
            boundaries,
            left,
            "left",
            args.closed,
            args.tangent_window,
        )
    )

    right_score = (
        three_side_orientation_score(
            boundaries,
            right,
            "right",
            args.closed,
            args.tangent_window,
        )
    )

    if left_score < 0.0:
        print(
            "[INFO] LEFT 진행방향이 MIDDLE과 반대 -> LEFT reverse"
        )

        left = reverse_polyline(
            left,
            args.closed,
        )

    if right_score < 0.0:
        print(
            "[INFO] RIGHT 진행방향이 MIDDLE과 반대 -> RIGHT reverse"
        )

        right = reverse_polyline(
            right,
            args.closed,
        )

    if (
        left_score < 0.0
        or
        right_score < 0.0
    ):
        populate_three_line_intersections(
            boundaries,
            left,
            right,
        )

    # --------------------------------------------------------
    # STEP 2: 초기 Polygon 생성
    # --------------------------------------------------------

    result = assemble_three_line_roi(
        args,
        left,
        middle,
        right,
        boundaries,
        frame_id,
    )

    # --------------------------------------------------------
    # STEP 3: 결과 화면에서 공유 경계 직접 드래그
    #
    # Enter를 눌러야 수정 결과가 확정된다.
    # 이 함수가 반환한 result가 이후 JSON/TXT/PNG로 저장된다.
    # --------------------------------------------------------

    result = (
        edit_three_line_boundaries_interactive(
            args,
            left,
            middle,
            right,
            boundaries,
            result,
            frame_id,
        )
    )

    return result


# ============================================================
# 10. Four-Line Pair ROI Builder
# ============================================================

def build_pair_roi(
    roi_name: str,
    reference_name: str,
    reference: np.ndarray,
    other_name: str,
    other: np.ndarray,
    all_lines_for_selection: Dict[
        str,
        np.ndarray,
    ],
    args,
    background_polygons: Optional[
        List[Dict]
    ] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[Dict],
    List[Dict],
    List[Dict],
]:
    reference_total = float(
        cumulative_lengths(
            reference
        )[-1]
    )

    boundaries = (
        select_pair_boundaries(
            roi_name=roi_name,
            reference_name=reference_name,
            reference=reference,
            other=other,
            all_lines=all_lines_for_selection,
            closed=args.closed,
            tangent_window=args.tangent_window,
            background_polygons=background_polygons,
        )
    )

    if not args.closed:
        boundaries = [
            make_pair_boundary_at_s(
                reference,
                0.0,
                False,
                args.tangent_window,
                "track_start",
            ),
            *boundaries,
            make_pair_boundary_at_s(
                reference,
                reference_total,
                False,
                args.tangent_window,
                "track_end",
            ),
        ]

    populate_pair_intersections(
        boundaries,
        other,
    )

    score = pair_orientation_score(
        boundaries,
        other,
        args.closed,
        args.tangent_window,
    )

    if score < 0.0:
        print(
            f"[INFO] {other_name.upper()} 진행방향이 "
            f"{reference_name.upper()}과 반대 -> reverse"
        )

        other = reverse_polyline(
            other,
            args.closed,
        )

        populate_pair_intersections(
            boundaries,
            other,
        )

    other_total = float(
        cumulative_lengths(
            other
        )[-1]
    )

    section_pairs = make_section_pairs(
        boundaries,
        args.closed,
    )

    section_types = make_section_types(
        len(section_pairs)
    )

    reference_s, reference_resampled = (
        resample_with_mandatory_s(
            reference,
            args.spacing,
            [
                boundary.reference_s
                for boundary
                in boundaries
            ],
            args.closed,
        )
    )

    other_s, other_resampled = (
        resample_with_mandatory_s(
            other,
            args.spacing,
            [
                boundary.other_s
                for boundary
                in boundaries
            ],
            args.closed,
        )
    )

    polygons = []
    sections = []

    straight_number = 0
    curve_number = 0

    for order_index, (
        boundary_pair,
        section_type,
    ) in enumerate(
        zip(
            section_pairs,
            section_types,
        ),
        start=1,
    ):
        start_boundary, end_boundary = (
            boundary_pair
        )

        if section_type == "STRAIGHT":
            straight_number += 1

            section_type_index = (
                straight_number
            )

            section_name = (
                f"straight"
                f"{section_type_index}"
            )

        else:
            curve_number += 1

            section_type_index = (
                curve_number
            )

            section_name = (
                f"curve"
                f"{section_type_index}"
            )

        reference_section = (
            extract_section_from_resampled(
                reference_s,
                reference_resampled,
                start_boundary.reference_s,
                end_boundary.reference_s,
                reference_total,
                args.closed,
            )
        )

        other_section = (
            extract_section_from_resampled(
                other_s,
                other_resampled,
                start_boundary.other_s,
                end_boundary.other_s,
                other_total,
                args.closed,
            )
        )

        vertices = close_polygon(
            np.vstack(
                [
                    other_section,
                    reference_section[::-1],
                ]
            )
        )

        polygon_id = (
            f"{roi_name}"
            f"{order_index}"
        )

        sections.append(
            {
                "roi": roi_name,
                "track_order": order_index,
                "section_type": section_type,
                "section_type_index": section_type_index,
                "section_name": section_name,
                "reference_line": reference_name,
                "other_line": other_name,
                "reference_points": points_to_list(
                    reference_section
                ),
                "other_points": points_to_list(
                    other_section
                ),
                "polygon_id": polygon_id,
            }
        )

        polygons.append(
            {
                "id": polygon_id,
                "roi": roi_name,
                "track_order": order_index,
                "section_type": section_type,
                "section_type_index": section_type_index,
                "section_name": section_name,
                "vertices": points_to_list(
                    vertices
                ),
            }
        )

    boundary_data = []

    for index, boundary in enumerate(
        boundaries,
        start=1,
    ):
        boundary_data.append(
            {
                "roi": roi_name,
                "id": index,
                "kind": boundary.kind,
                "reference_line": reference_name,
                "other_line": other_name,
                "reference_s": float(
                    boundary.reference_s
                ),
                "other_s": float(
                    boundary.other_s
                ),
                "points": [
                    points_to_list(
                        [boundary.other_point]
                    )[0],
                    points_to_list(
                        [boundary.reference_point]
                    )[0],
                ],
            }
        )

    return (
        reference_resampled,
        other_resampled,
        sections,
        polygons,
        boundary_data,
    )


# ============================================================
# 11. Four-Line ROI Build
# ============================================================

def build_four_line_roi(
    args,
    data_dir: str,
) -> Dict:
    file_map = {
        "left": os.path.join(
            data_dir,
            "left_raw.csv",
        ),
        "middle1": os.path.join(
            data_dir,
            "middle1_raw.csv",
        ),
        "middle2": os.path.join(
            data_dir,
            "middle2_raw.csv",
        ),
        "right": os.path.join(
            data_dir,
            "right_raw.csv",
        ),
    }

    raw_lines = {}
    frame_ids = []

    for name, path in file_map.items():
        points, frame_id = load_line_csv(
            path
        )

        raw_lines[name] = (
            prepare_polyline(
                points,
                args.closed,
            )
        )

        if frame_id:
            frame_ids.append(
                frame_id
            )

    if (
        frame_ids
        and
        len(
            set(frame_ids)
        )
        != 1
    ):
        raise ValueError(
            "LEFT/MIDDLE1/MIDDLE2/RIGHT의 frame_id가 서로 다릅니다."
        )

    left = raw_lines["left"]
    middle1 = raw_lines["middle1"]
    middle2 = raw_lines["middle2"]
    right = raw_lines["right"]

    all_lines = {
        "left": left,
        "middle1": middle1,
        "middle2": middle2,
        "right": right,
    }

    # --------------------------------------------------------
    # STEP 1: IN 한 바퀴 클릭
    # IN = LEFT <-> MIDDLE1
    # --------------------------------------------------------

    (
        middle1_resampled,
        left_resampled,
        in_sections,
        in_polygons,
        in_boundaries,
    ) = build_pair_roi(
        roi_name="IN",
        reference_name="middle1",
        reference=middle1,
        other_name="left",
        other=left,
        all_lines_for_selection=all_lines,
        args=args,
        background_polygons=None,
    )

    print(
        "\n[OK] IN Polygon 생성 완료."
    )
    print(
        "다음 창에서 생성된 IN Polygon을 확인한 뒤 "
        "MIDDLE2 위에 OUT 경계점을 한 바퀴 클릭하십시오."
    )

    # --------------------------------------------------------
    # STEP 2: IN Polygon을 띄운 상태에서 OUT 클릭
    # OUT = MIDDLE2 <-> RIGHT
    # --------------------------------------------------------

    (
        middle2_resampled,
        right_resampled,
        out_sections,
        out_polygons,
        out_boundaries,
    ) = build_pair_roi(
        roi_name="OUT",
        reference_name="middle2",
        reference=middle2,
        other_name="right",
        other=right,
        all_lines_for_selection=all_lines,
        args=args,
        background_polygons=in_polygons,
    )

    polygons = (
        in_polygons
        +
        out_polygons
    )

    boundaries = (
        in_boundaries
        +
        out_boundaries
    )

    return {
        "metadata": {
            "coordinate_source": "/odom pose.position.x/y",
            "line_count": 4,
            "line_mode": "LEFT / MIDDLE1 / MIDDLE2 / RIGHT",
            "frame_id": (
                frame_ids[0]
                if frame_ids
                else ""
            ),
            "closed_track": bool(
                args.closed
            ),
            "resample_spacing_m": float(
                args.spacing
            ),
            "tangent_window_m": float(
                args.tangent_window
            ),
            "roi_definition": {
                "IN": "LEFT <-> MIDDLE1",
                "CENTER_GAP": "MIDDLE1 <-> MIDDLE2 (ROI 생성 안 함)",
                "OUT": "MIDDLE2 <-> RIGHT",
            },
        },
        "resampled_lines": {
            "left": points_to_list(
                left_resampled
            ),
            "middle1": points_to_list(
                middle1_resampled
            ),
            "middle2": points_to_list(
                middle2_resampled
            ),
            "right": points_to_list(
                right_resampled
            ),
        },
        "boundaries": boundaries,
        "roi_sections": {
            "IN": in_sections,
            "OUT": out_sections,
        },
        "polygons": polygons,
    }


# ============================================================
# 12. Main ROI Build
# ============================================================

def build_track_roi(
    args,
):
    data_dir = os.path.abspath(
        args.data_dir
    )

    line_count = (
        ask_line_count()
    )

    print(
        "\n============================================"
    )
    print(
        f"JSON output : {args.output}"
    )
    print(
        f"TXT output  : {args.txt_output}"
    )
    print(
        f"Figure      : {args.figure}"
    )

    if line_count == 3:
        print(
            "3-Line 모드 선택"
        )
        print(
            "읽는 파일:"
        )
        print(
            "  left_raw.csv"
        )
        print(
            "  middle_raw.csv"
        )
        print(
            "  right_raw.csv"
        )

    else:
        print(
            "4-Line 모드 선택"
        )
        print(
            "읽는 파일:"
        )
        print(
            "  left_raw.csv"
        )
        print(
            "  middle1_raw.csv"
        )
        print(
            "  middle2_raw.csv"
        )
        print(
            "  right_raw.csv"
        )

    print(
        "============================================"
    )

    if line_count == 3:
        result = build_three_line_roi(
            args,
            data_dir,
        )

    else:
        result = build_four_line_roi(
            args,
            data_dir,
        )

    save_json(
        result,
        args.output,
    )

    # C++ ROI node용 Polygon TXT 저장
    save_roi_txt(
        result,
        args.txt_output,
    )

    # LiDAR filtering용 Cell TXT 저장
    save_roi_cells_txt(
        result,
        args.cells_output,
        args.spacing,
    )

    print(
        f"\n[OK] ROI JSON 저장: "
        f"{args.output}"
    )

    print(
        f"[OK] ROI TXT 저장 : "
        f"{args.txt_output}"
    )

    print(
        f"[OK] CELL TXT 저장: "
        f"{args.cells_output}"
    )

    print(
        f"[OK] IN Polygon: "
        f"{len(result['roi_sections']['IN'])}개"
    )

    print(
        f"[OK] OUT Polygon: "
        f"{len(result['roi_sections']['OUT'])}개"
    )

    visualize_roi_data(
        result,
        args.figure,
        show=not args.no_show,
    )


# ============================================================
# 13. Final Visualization
# ============================================================



def draw_cell_overlay(
    ax,
    data: Dict,
):
    """
    최종 시각화에서 ROI Cell 분할선까지 함께 표시한다.

    - IN Cells  : dark green line
    - OUT Cells : dark blue line

    Cell 정보는 JSON에 직접 저장하지 않고,
    roi_sections + resample_spacing_m 정보를 이용해
    시각화 시 다시 생성한다.
    """
    metadata = data.get(
        "metadata",
        {},
    )

    roi_sections = data.get(
        "roi_sections",
        {},
    )

    spacing = float(
        metadata.get(
            "resample_spacing_m",
            0.5,
        )
    )

    if spacing <= 0.0:
        return

    try:
        in_cells = build_cells_from_sections(
            roi_sections.get(
                "IN",
                [],
            ),
            spacing,
        )

        out_cells = build_cells_from_sections(
            roi_sections.get(
                "OUT",
                [],
            ),
            spacing,
        )

    except Exception as exc:
        print(
            f"[WARN] Cell 시각화 생성 실패: {exc}"
        )
        return

    first_in = True
    first_out = True

    for vertices in in_cells:
        vertices_array = np.asarray(
            vertices,
            dtype=float,
        )

        patch = MplPolygon(
            vertices_array,
            closed=True,
            facecolor="none",
            edgecolor="darkgreen",
            linewidth=0.55,
            alpha=0.75,
            zorder=6,
            label=(
                "IN cells"
                if first_in
                else None
            ),
        )

        ax.add_patch(
            patch
        )

        first_in = False

    for vertices in out_cells:
        vertices_array = np.asarray(
            vertices,
            dtype=float,
        )

        patch = MplPolygon(
            vertices_array,
            closed=True,
            facecolor="none",
            edgecolor="darkblue",
            linewidth=0.55,
            alpha=0.75,
            zorder=6,
            label=(
                "OUT cells"
                if first_out
                else None
            ),
        )

        ax.add_patch(
            patch
        )

        first_out = False

def visualize_roi_data(
    data: Dict,
    save_path: str = None,
    show: bool = True,
):
    fig, ax = plt.subplots(
        figsize=(14, 10)
    )

    draw_polygon_list(
        ax,
        data["polygons"],
        show_label=True,
    )

    lines = {
        name: np.asarray(
            points,
            dtype=float,
        )
        for name, points
        in data[
            "resampled_lines"
        ].items()
    }

    draw_track_lines(
        ax,
        lines,
        scatter=True,
    )

    # --------------------------------------------------------
    # Cell overlay
    # track_roi_cells.txt와 동일한 Cell 구성을
    # 최종 Figure에도 함께 시각화한다.
    # --------------------------------------------------------
    draw_cell_overlay(
        ax,
        data,
    )

    first_shared = True
    first_in = True
    first_out = True

    for boundary in data[
        "boundaries"
    ]:
        points = np.asarray(
            boundary["points"],
            dtype=float,
        )

        roi = boundary["roi"]

        if roi == "SHARED":
            color = "crimson"

            label = (
                "Shared section boundary"
                if first_shared
                else None
            )

            first_shared = False

        elif roi == "IN":
            color = "royalblue"

            label = (
                "IN section boundary"
                if first_in
                else None
            )

            first_in = False

        else:
            color = "purple"

            label = (
                "OUT section boundary"
                if first_out
                else None
            )

            first_out = False

        ax.plot(
            points[:, 0],
            points[:, 1],
            "--",
            color=color,
            linewidth=1.0,
            alpha=0.85,
            label=label,
        )

        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=24,
            color=color,
            zorder=5,
        )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True
    )

    ax.set_xlabel(
        "x [m]"
    )

    ax.set_ylabel(
        "y [m]"
    )

    if (
        data["metadata"]["line_count"]
        ==
        3
    ):
        title = (
            "3-Line Track ROI Polygons"
        )

    else:
        title = (
            "4-Line Track ROI Polygons "
            "(MIDDLE1-MIDDLE2 gap excluded)"
        )

    ax.set_title(
        title
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    if save_path:
        os.makedirs(
            os.path.dirname(
                os.path.abspath(
                    save_path
                )
            ),
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

        print(
            f"[OK] 시각화 이미지 저장: "
            f"{save_path}"
        )

    if show:
        plt.show()

    else:
        plt.close(
            fig
        )


def visualize_from_json(
    args,
):
    with open(
        args.input,
        "r",
        encoding="utf-8",
    ) as fp:
        data = json.load(
            fp
        )

    visualize_roi_data(
        data,
        args.save,
        show=not args.no_show,
    )


# ============================================================
# 14. CLI
# ============================================================

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "3-Line 또는 4-Line 차선 경계 데이터로 "
            "IN/OUT Polygon을 생성하는 Track ROI Builder"
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # --------------------------------------------------------
    # record
    # --------------------------------------------------------

    record_parser = sub.add_parser(
        "record",
        help="ROS2 /odom x,y를 차선 경계 데이터로 기록",
    )

    record_parser.add_argument(
        "--line",
        required=True,
        choices=[
            "left",
            "middle",
            "middle1",
            "middle2",
            "right",
        ],
        help=(
            "3-Line: left/middle/right, "
            "4-Line: left/middle1/middle2/right"
        ),
    )

    record_parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=(
            "기록 데이터 저장 폴더. "
            f"기본값: {DEFAULT_DATA_DIR}"
        ),
    )

    record_parser.add_argument(
        "--topic",
        default="/odom",
    )

    record_parser.add_argument(
        "--min-distance",
        type=float,
        default=0.05,
        help="연속 기록점 최소 거리 [m] (기본 0.05)",
    )

    # --------------------------------------------------------
    # build
    # --------------------------------------------------------

    build_parser = sub.add_parser(
        "build",
        help=(
            "차선 개수 선택 -> 구간 클릭 -> "
            "재샘플링 -> Polygon 생성 -> JSON 저장"
        ),
    )

    build_parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=(
            "입력 차선 데이터 폴더. "
            f"기본값: {DEFAULT_DATA_DIR}"
        ),
    )

    build_parser.add_argument(
        "--spacing",
        type=float,
        required=True,
        help=(
            "재샘플링 간격 [m]. "
            "구간 경계점은 spacing과 무관하게 추가됩니다."
        ),
    )

    build_parser.add_argument(
        "--closed",
        action="store_true",
        help=(
            "폐곡선 트랙이면 지정. "
            "미지정 시 open track."
        ),
    )

    build_parser.add_argument(
        "--tangent-window",
        type=float,
        default=1.0,
        help=(
            "기준선 접선 계산 범위 [m] "
            "(기본 1.0)"
        ),
    )

    build_parser.add_argument(
        "--output",
        default=DEFAULT_JSON_PATH,
        help=(
            "JSON 저장 경로. "
            f"기본값: {DEFAULT_JSON_PATH}"
        ),
    )

    build_parser.add_argument(
        "--txt-output",
        default=DEFAULT_TXT_PATH,
        help=(
            "C++ ROI node용 TXT 저장 경로. "
            f"기본값: {DEFAULT_TXT_PATH}"
        ),
    )

    build_parser.add_argument(
        "--figure",
        default=DEFAULT_FIGURE_PATH,
        help=(
            "최종 ROI PNG 저장 경로. "
            f"기본값: {DEFAULT_FIGURE_PATH}"
        ),
    )

    build_parser.add_argument(
        "--cells-output",
        default=DEFAULT_CELLS_PATH,
        help=(
            "LiDAR ROI filtering용 Cell TXT 저장 경로."
        ),
    )

    build_parser.add_argument(
        "--no-show",
        action="store_true",
    )

    # --------------------------------------------------------
    # visualize
    # --------------------------------------------------------

    visualize_parser = sub.add_parser(
        "visualize",
        help="생성된 track_roi.json 재시각화",
    )

    visualize_parser.add_argument(
        "--input",
        default=DEFAULT_JSON_PATH,
        help=(
            "시각화할 JSON 파일. "
            f"기본값: {DEFAULT_JSON_PATH}"
        ),
    )

    visualize_parser.add_argument(
        "--save",
        default=None,
    )

    visualize_parser.add_argument(
        "--no-show",
        action="store_true",
    )

    return parser


# ============================================================
# 15. Main
# ============================================================

def main():
    parser = build_arg_parser()

    # 인자 없이 실행하면:
    # python3 track_roi_builder.py build --spacing 0.5 --closed
    # 와 동일하게 동작한다.
    if len(sys.argv) == 1:
        args = parser.parse_args(
            [
                "build",
                "--spacing",
                "0.5",
                "--closed",
            ]
        )
    else:
        args = parser.parse_args()

    try:
        if args.command == "record":
            output_path = os.path.join(
                os.path.abspath(
                    args.data_dir
                ),
                f"{args.line}_raw.csv",
            )

            recorder = OdomLineRecorder(
                line_name=args.line,
                output_path=output_path,
                topic=args.topic,
                min_distance=args.min_distance,
            )

            recorder.run()

        elif args.command == "build":
            if args.spacing <= 0.0:
                raise ValueError(
                    "--spacing은 0보다 커야 합니다."
                )

            if args.tangent_window <= 0.0:
                raise ValueError(
                    "--tangent-window는 0보다 커야 합니다."
                )

            build_track_roi(
                args
            )

        elif args.command == "visualize":
            visualize_from_json(
                args
            )

    except KeyboardInterrupt:
        print(
            "\n사용자에 의해 중단되었습니다."
        )

        sys.exit(
            130
        )

    except Exception as exc:
        print(
            f"\n[ERROR] {exc}",
            file=sys.stderr,
        )

        sys.exit(
            1
        )


if __name__ == "__main__":
    main()
