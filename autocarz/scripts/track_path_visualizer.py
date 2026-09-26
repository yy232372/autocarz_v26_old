#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
track_path_visualizer.py

============================================================
기능
============================================================

Track / Path / Circular ROI 시각화.

지원 방식:

1. PATH IN만
2. PATH OUT만
3. PATH IN + PATH OUT
4. Track + PATH IN
5. Track + PATH OUT
6. Track + PATH IN + PATH OUT

Track JSON:
    입력하지 않고 Enter -> Track 표시 안 함

PATH IN:
    입력하지 않고 Enter -> PATH IN 표시 안 함

PATH OUT:
    입력하지 않고 Enter -> PATH OUT 표시 안 함

단:
    PATH IN / PATH OUT 둘 다 없으면 실행하지 않는다.

이미지 저장:
    Enter -> 저장 안 함
"""

import json
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection


# ============================================================
# 1. 기본 경로
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PACKAGE_DIR = os.path.dirname(
    SCRIPT_DIR
)

PATH_DIR = os.path.join(
    PACKAGE_DIR,
    "path"
)

TRACK_DATA_DIR = os.path.join(
    PATH_DIR,
    "track_data"
)


# ============================================================
# 2. 기본 설정
# ============================================================

DEFAULT_PLACE = "CBuniv"

DEFAULT_ROUND = "본선"


# ============================================================
# 3. 장소 입력
# ============================================================

def ask_place():

    value = input(
        f"장소 코드 [{DEFAULT_PLACE}]: "
    ).strip()

    if value == "":
        value = DEFAULT_PLACE

    return value


# ============================================================
# 4. 본선 / 예선
# ============================================================

def ask_round():

    while True:

        value = input(
            f"구분 (본선/예선) [{DEFAULT_ROUND}]: "
        ).strip()

        if value == "":
            value = DEFAULT_ROUND

        normalized = value.lower()

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


# ============================================================
# 5. Event Path 폴더
# ============================================================

def get_event_path_dir(
    place,
    round_suffix
):

    folder_name = (
        f"{place}_{round_suffix}"
    )

    return os.path.abspath(
        os.path.join(
            PATH_DIR,
            folder_name
        )
    )


# ============================================================
# 6. Track JSON 입력
# ============================================================

def ask_track_json_path():

    value = input(
        "Track ROI JSON 파일 (없으면 Enter): "
    ).strip()

    if value == "":
        return None

    if os.path.dirname(
        value
    ):

        return os.path.abspath(
            value
        )

    return os.path.abspath(
        os.path.join(
            TRACK_DATA_DIR,
            value
        )
    )


# ============================================================
# 7. Path 파일 입력
# ============================================================

def ask_optional_path_file(
    message,
    event_path_dir
):
    """
    Enter:
        해당 Path 사용 안 함

    파일명:
        event_path_dir에서 검색

    전체/상대 경로:
        입력 경로 사용
    """

    value = input(
        f"{message} (없으면 Enter): "
    ).strip()

    if value == "":
        return None

    if os.path.dirname(
        value
    ):

        return os.path.abspath(
            value
        )

    return os.path.abspath(
        os.path.join(
            event_path_dir,
            value
        )
    )


# ============================================================
# 8. 이미지 저장 여부
# ============================================================

def ask_output_image(
    event_path_dir
):

    value = input(
        "이미지 저장 파일명 (저장 안 하면 Enter): "
    ).strip()

    if value == "":
        return None

    if not value.lower().endswith(
        (
            ".png",
            ".jpg",
            ".jpeg"
        )
    ):

        value += ".png"

    if os.path.dirname(
        value
    ):

        return os.path.abspath(
            value
        )

    return os.path.abspath(
        os.path.join(
            event_path_dir,
            value
        )
    )


# ============================================================
# 9. Track JSON 로드
# ============================================================

def load_track_json(
    file_path
):

    if not os.path.exists(
        file_path
    ):

        raise FileNotFoundError(
            "Track JSON 파일을 찾을 수 없습니다:\n"
            f"{file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as fp:

        data = json.load(
            fp
        )

    if "resampled_lines" not in data:

        raise KeyError(
            "JSON에 'resampled_lines'가 없습니다."
        )

    lines = data[
        "resampled_lines"
    ]

    metadata = data.get(
        "metadata",
        {}
    )

    line_count = metadata.get(
        "line_count",
        None
    )

    if line_count is None:

        if (
            "left" in lines
            and
            "middle" in lines
            and
            "right" in lines
        ):

            line_count = 3

        elif (
            "left" in lines
            and
            "middle1" in lines
            and
            "middle2" in lines
            and
            "right" in lines
        ):

            line_count = 4

        else:

            raise ValueError(
                "3-Line / 4-Line 구조를 판단할 수 없습니다."
            )

    line_count = int(
        line_count
    )

    if line_count not in [
        3,
        4
    ]:

        raise ValueError(
            f"지원하지 않는 line_count: {line_count}"
        )

    return (
        data,
        line_count
    )


# ============================================================
# 10. Track 좌표 변환
# ============================================================

def convert_resampled_lines(
    data,
    line_count
):

    source = data[
        "resampled_lines"
    ]

    if line_count == 3:

        required_names = [
            "left",
            "middle",
            "right"
        ]

    else:

        required_names = [
            "left",
            "middle1",
            "middle2",
            "right"
        ]

    lines = {}

    for name in required_names:

        if name not in source:

            raise KeyError(
                f"resampled_lines에 '{name}'가 없습니다."
            )

        points = np.asarray(
            source[name],
            dtype=float
        )

        if (
            points.ndim != 2
            or
            points.shape[1] < 2
        ):

            raise ValueError(
                f"{name} 데이터 구조가 올바르지 않습니다.\n"
                f"shape = {points.shape}"
            )

        points = points[
            :,
            :2
        ]

        if len(
            points
        ) < 2:

            raise ValueError(
                f"{name} 좌표가 2개 미만입니다."
            )

        lines[
            name
        ] = points

    return lines


# ============================================================
# 11. Path TXT 로드
# ============================================================

def load_path_txt(
    file_path
):

    if not os.path.exists(
        file_path
    ):

        raise FileNotFoundError(
            "Path 파일을 찾을 수 없습니다:\n"
            f"{file_path}"
        )

    points = []

    radii = []

    with open(
        file_path,
        "r",
        encoding="utf-8-sig"
    ) as fp:

        for line_number, line in enumerate(
            fp,
            start=1
        ):

            line = line.strip()

            if line == "":
                continue

            if line.startswith(
                "#"
            ):

                continue

            line = line.replace(
                ",",
                " "
            )

            values = line.split()

            if len(
                values
            ) < 2:

                continue

            try:

                x = float(
                    values[0]
                )

                y = float(
                    values[1]
                )

            except ValueError:

                print(
                    f"[WARNING] "
                    f"{os.path.basename(file_path)} "
                    f"{line_number}번째 줄 제외"
                )

                continue

            if (
                not math.isfinite(
                    x
                )
                or
                not math.isfinite(
                    y
                )
            ):

                continue

            radius = np.nan

            if len(
                values
            ) >= 4:

                try:

                    radius = float(
                        values[3]
                    )

                except ValueError:

                    radius = np.nan

            points.append(
                [
                    x,
                    y
                ]
            )

            radii.append(
                radius
            )

    if len(
        points
    ) < 2:

        raise ValueError(
            f"{file_path}\n"
            "유효한 waypoint가 2개 미만입니다."
        )

    return (
        np.asarray(
            points,
            dtype=float
        ),
        np.asarray(
            radii,
            dtype=float
        )
    )


# ============================================================
# 12. Path 길이
# ============================================================

def calculate_path_length(
    points
):

    if (
        points is None
        or
        len(points) < 2
    ):

        return 0.0

    distances = np.linalg.norm(
        np.diff(
            points,
            axis=0
        ),
        axis=1
    )

    return float(
        np.sum(
            distances
        )
    )


# ============================================================
# 13. Waypoint 간격
# ============================================================

def calculate_spacing(
    points
):

    if (
        points is None
        or
        len(points) < 2
    ):

        return (
            0.0,
            0.0,
            0.0
        )

    distances = np.linalg.norm(
        np.diff(
            points,
            axis=0
        ),
        axis=1
    )

    return (
        float(
            np.mean(
                distances
            )
        ),
        float(
            np.min(
                distances
            )
        ),
        float(
            np.max(
                distances
            )
        )
    )


# ============================================================
# 14. 원형 ROI
# ============================================================

def draw_circular_roi(
    ax,
    points,
    radii,
    edge_color,
    face_color,
    label
):

    if points is None:
        return

    if radii is None:
        return

    fill_circles = []

    edge_circles = []

    valid_count = 0

    for point, radius in zip(
        points,
        radii
    ):

        if not np.isfinite(
            radius
        ):

            continue

        if radius <= 0.0:

            continue

        fill_circles.append(
            Circle(
                (
                    point[0],
                    point[1]
                ),
                radius=radius
            )
        )

        edge_circles.append(
            Circle(
                (
                    point[0],
                    point[1]
                ),
                radius=radius
            )
        )

        valid_count += 1

    if valid_count == 0:

        print(
            f"[WARNING] {label}: "
            "유효한 ROI radius가 없습니다."
        )

        return

    # ROI 내부
    fill_collection = PatchCollection(
        fill_circles,
        facecolor=face_color,
        edgecolor="none",
        alpha=0.045,
        zorder=4
    )

    ax.add_collection(
        fill_collection
    )

    # ROI 경계
    edge_collection = PatchCollection(
        edge_circles,
        facecolor="none",
        edgecolor=edge_color,
        linewidth=0.6,
        alpha=0.22,
        zorder=5
    )

    edge_collection.set_label(
        label
    )

    ax.add_collection(
        edge_collection
    )

    print(
        f"{label} : {valid_count} circles"
    )


# ============================================================
# 15. 3-Line Track 영역
# ============================================================

def fill_track_three_line(
    ax,
    lines
):

    left = lines[
        "left"
    ]

    middle = lines[
        "middle"
    ]

    right = lines[
        "right"
    ]

    # IN
    count = min(
        len(left),
        len(middle)
    )

    x = np.concatenate(
        [
            left[:count, 0],
            middle[:count, 0][::-1]
        ]
    )

    y = np.concatenate(
        [
            left[:count, 1],
            middle[:count, 1][::-1]
        ]
    )

    ax.fill(
        x,
        y,
        alpha=0.12,
        label="IN Track"
    )

    # OUT
    count = min(
        len(middle),
        len(right)
    )

    x = np.concatenate(
        [
            middle[:count, 0],
            right[:count, 0][::-1]
        ]
    )

    y = np.concatenate(
        [
            middle[:count, 1],
            right[:count, 1][::-1]
        ]
    )

    ax.fill(
        x,
        y,
        alpha=0.12,
        label="OUT Track"
    )


# ============================================================
# 16. 4-Line Track 영역
# ============================================================

def fill_track_four_line(
    ax,
    lines
):

    left = lines[
        "left"
    ]

    middle1 = lines[
        "middle1"
    ]

    middle2 = lines[
        "middle2"
    ]

    right = lines[
        "right"
    ]

    # IN
    count = min(
        len(left),
        len(middle1)
    )

    x = np.concatenate(
        [
            left[:count, 0],
            middle1[:count, 0][::-1]
        ]
    )

    y = np.concatenate(
        [
            left[:count, 1],
            middle1[:count, 1][::-1]
        ]
    )

    ax.fill(
        x,
        y,
        alpha=0.12,
        label="IN Track"
    )

    # OUT
    count = min(
        len(middle2),
        len(right)
    )

    x = np.concatenate(
        [
            middle2[:count, 0],
            right[:count, 0][::-1]
        ]
    )

    y = np.concatenate(
        [
            middle2[:count, 1],
            right[:count, 1][::-1]
        ]
    )

    ax.fill(
        x,
        y,
        alpha=0.12,
        label="OUT Track"
    )


# ============================================================
# 17. Track Line
# ============================================================

def draw_track_lines(
    ax,
    lines
):

    colors = {
        "left": "black",
        "middle": "darkorange",
        "middle1": "darkorange",
        "middle2": "goldenrod",
        "right": "dimgray"
    }

    for name, points in lines.items():

        color = colors.get(
            name,
            "black"
        )

        ax.plot(
            points[:, 0],
            points[:, 1],
            color=color,
            linewidth=1.8,
            label=name.upper(),
            zorder=7
        )

        ax.scatter(
            points[:, 0],
            points[:, 1],
            color=color,
            s=6,
            alpha=0.4,
            zorder=7
        )


# ============================================================
# 18. 단일 Path 표시
# ============================================================

def draw_single_path(
    ax,
    points,
    color,
    label
):

    if points is None:
        return

    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linewidth=2.5,
        label=label,
        zorder=10
    )

    ax.scatter(
        points[:, 0],
        points[:, 1],
        color=color,
        s=14,
        zorder=11
    )

    # 시작점
    ax.scatter(
        points[0, 0],
        points[0, 1],
        marker="*",
        s=130,
        color=color,
        edgecolor="black",
        linewidth=0.7,
        label=f"{label} Start",
        zorder=12
    )


# ============================================================
# 19. 시각화
# ============================================================

def visualize(
    lines,
    line_count,
    path_in,
    path_out,
    path_in_radius,
    path_out_radius,
    output_image
):

    fig, ax = plt.subplots(
        figsize=(
            14,
            10
        )
    )

    # --------------------------------------------------------
    # Track 영역
    # --------------------------------------------------------

    if (
        lines is not None
        and
        line_count is not None
    ):

        if line_count == 3:

            fill_track_three_line(
                ax,
                lines
            )

        else:

            fill_track_four_line(
                ax,
                lines
            )

    # --------------------------------------------------------
    # Circular ROI
    # --------------------------------------------------------

    print()
    print(
        "------------------------------------------------------------"
    )

    print(
        "CIRCULAR ROI"
    )

    print(
        "------------------------------------------------------------"
    )

    if path_in is not None:

        draw_circular_roi(
            ax=ax,
            points=path_in,
            radii=path_in_radius,
            edge_color="royalblue",
            face_color="royalblue",
            label="PATH IN Circular ROI"
        )

    if path_out is not None:

        draw_circular_roi(
            ax=ax,
            points=path_out,
            radii=path_out_radius,
            edge_color="crimson",
            face_color="crimson",
            label="PATH OUT Circular ROI"
        )

    # --------------------------------------------------------
    # Track Line
    # --------------------------------------------------------

    if lines is not None:

        draw_track_lines(
            ax,
            lines
        )

    # --------------------------------------------------------
    # PATH IN
    # --------------------------------------------------------

    draw_single_path(
        ax=ax,
        points=path_in,
        color="royalblue",
        label="PATH IN"
    )

    # --------------------------------------------------------
    # PATH OUT
    # --------------------------------------------------------

    draw_single_path(
        ax=ax,
        points=path_out,
        color="crimson",
        label="PATH OUT"
    )

    # --------------------------------------------------------
    # Figure 설정
    # --------------------------------------------------------

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.set_xlabel(
        "x [m]"
    )

    ax.set_ylabel(
        "y [m]"
    )

    # 제목
    path_count = int(
        path_in is not None
    ) + int(
        path_out is not None
    )

    if lines is not None:

        ax.set_title(
            f"{line_count}-Line Track + {path_count} Path"
        )

    else:

        if path_count == 1:

            ax.set_title(
                "Single Path + Circular ROI"
            )

        else:

            ax.set_title(
                "PATH IN + PATH OUT + Circular ROI"
            )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    # --------------------------------------------------------
    # 이미지 저장
    # --------------------------------------------------------

    if output_image is not None:

        output_directory = os.path.dirname(
            output_image
        )

        if output_directory:

            os.makedirs(
                output_directory,
                exist_ok=True
            )

        fig.savefig(
            output_image,
            dpi=200,
            bbox_inches="tight"
        )

        print()
        print(
            "[OK] 이미지 저장:"
        )

        print(
            output_image
        )

    else:

        print()
        print(
            "[INFO] 이미지 저장 안 함"
        )

    # --------------------------------------------------------
    # 화면 표시
    # --------------------------------------------------------

    plt.show()


# ============================================================
# 20. 단일 Path 정보 출력
# ============================================================

def print_single_path_information(
    label,
    points,
    radii
):

    if points is None:

        print()
        print(
            f"{label} : 사용 안 함"
        )

        return

    mean_spacing, min_spacing, max_spacing = calculate_spacing(
        points
    )

    print()
    print(
        "------------------------------------------------------------"
    )

    print(
        label
    )

    print(
        "------------------------------------------------------------"
    )

    print(
        f"Waypoint : {len(points)}"
    )

    print(
        f"Length   : "
        f"{calculate_path_length(points):.3f} m"
    )

    print(
        "Spacing  : "
        f"mean={mean_spacing:.3f} m / "
        f"min={min_spacing:.3f} m / "
        f"max={max_spacing:.3f} m"
    )

    if radii is None:
        return

    valid_radius = radii[
        np.isfinite(
            radii
        )
        &
        (
            radii > 0.0
        )
    ]

    if len(
        valid_radius
    ) > 0:

        print(
            "Radius   : "
            f"mean={np.mean(valid_radius):.3f} m / "
            f"min={np.min(valid_radius):.3f} m / "
            f"max={np.max(valid_radius):.3f} m"
        )

    else:

        print(
            "Radius   : 없음"
        )


# ============================================================
# 21. 전체 정보 출력
# ============================================================

def print_information(
    data,
    lines,
    line_count,
    path_in,
    path_out,
    path_in_radius,
    path_out_radius
):

    print()
    print(
        "============================================================"
    )

    print(
        "VISUALIZATION INFORMATION"
    )

    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Track 정보
    # --------------------------------------------------------

    if (
        data is not None
        and
        lines is not None
        and
        line_count is not None
    ):

        print(
            f"Track Mode : {line_count}-Line"
        )

        metadata = data.get(
            "metadata",
            {}
        )

        if "resample_spacing_m" in metadata:

            print(
                "Track resample spacing : "
                f"{metadata['resample_spacing_m']} m"
            )

        print()

        for name, points in lines.items():

            print(
                f"{name:8s} : "
                f"{len(points):5d} points / "
                f"{calculate_path_length(points):.3f} m"
            )

    else:

        print(
            "Track Mode : 사용 안 함"
        )

    # --------------------------------------------------------
    # Path 정보
    # --------------------------------------------------------

    print_single_path_information(
        label="PATH IN",
        points=path_in,
        radii=path_in_radius
    )

    print_single_path_information(
        label="PATH OUT",
        points=path_out,
        radii=path_out_radius
    )

    print()
    print(
        "============================================================"
    )


# ============================================================
# 22. Main
# ============================================================

def main():

    print()
    print(
        "============================================================"
    )

    print(
        "TRACK / PATH / CIRCULAR ROI VISUALIZER"
    )

    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # 장소
    # --------------------------------------------------------

    place = ask_place()

    # --------------------------------------------------------
    # 본선 / 예선
    # --------------------------------------------------------

    (
        round_name,
        round_suffix
    ) = ask_round()

    # --------------------------------------------------------
    # Path 폴더
    # --------------------------------------------------------

    event_path_dir = get_event_path_dir(
        place,
        round_suffix
    )

    if not os.path.isdir(
        event_path_dir
    ):

        raise FileNotFoundError(
            "선택한 Path 폴더를 찾을 수 없습니다:\n"
            f"{event_path_dir}\n\n"
            f"장소 = {place}\n"
            f"구분 = {round_name}"
        )

    print()
    print(
        "---------------- 경로 설정 ----------------"
    )

    print(
        f"장소             : {place}"
    )

    print(
        f"구분             : {round_name}"
    )

    print(
        f"Track 검색 폴더  : {TRACK_DATA_DIR}"
    )

    print(
        f"Path 검색 폴더   : {event_path_dir}"
    )

    print(
        "-------------------------------------------"
    )

    # --------------------------------------------------------
    # Track JSON
    # --------------------------------------------------------

    json_file = ask_track_json_path()

    # --------------------------------------------------------
    # PATH IN
    # Enter 가능
    # --------------------------------------------------------

    path_in_file = ask_optional_path_file(
        "PATH IN 파일",
        event_path_dir
    )

    # --------------------------------------------------------
    # PATH OUT
    # Enter 가능
    # --------------------------------------------------------

    path_out_file = ask_optional_path_file(
        "PATH OUT 파일",
        event_path_dir
    )

    # --------------------------------------------------------
    # Path가 하나도 없는 경우
    # --------------------------------------------------------

    if (
        path_in_file is None
        and
        path_out_file is None
    ):

        raise ValueError(
            "PATH IN / PATH OUT 중 최소 하나는 입력해야 합니다."
        )

    # --------------------------------------------------------
    # 이미지 저장
    # --------------------------------------------------------

    output_image = ask_output_image(
        event_path_dir
    )

    # --------------------------------------------------------
    # 입력 정보 출력
    # --------------------------------------------------------

    print()
    print(
        "---------------- 입력 파일 ----------------"
    )

    if json_file is None:

        print(
            "Track JSON : 사용 안 함"
        )

    else:

        print(
            f"Track JSON : {json_file}"
        )

    if path_in_file is None:

        print(
            "PATH IN    : 사용 안 함"
        )

    else:

        print(
            f"PATH IN    : {path_in_file}"
        )

    if path_out_file is None:

        print(
            "PATH OUT   : 사용 안 함"
        )

    else:

        print(
            f"PATH OUT   : {path_out_file}"
        )

    if output_image is None:

        print(
            "Image      : 저장 안 함"
        )

    else:

        print(
            f"Image      : {output_image}"
        )

    print(
        "-------------------------------------------"
    )

    # --------------------------------------------------------
    # Track
    # --------------------------------------------------------

    data = None

    lines = None

    line_count = None

    if json_file is not None:

        (
            data,
            line_count
        ) = load_track_json(
            json_file
        )

        lines = convert_resampled_lines(
            data,
            line_count
        )

    # --------------------------------------------------------
    # PATH IN
    # --------------------------------------------------------

    path_in = None

    path_in_radius = None

    if path_in_file is not None:

        (
            path_in,
            path_in_radius
        ) = load_path_txt(
            path_in_file
        )

    # --------------------------------------------------------
    # PATH OUT
    # --------------------------------------------------------

    path_out = None

    path_out_radius = None

    if path_out_file is not None:

        (
            path_out,
            path_out_radius
        ) = load_path_txt(
            path_out_file
        )

    # --------------------------------------------------------
    # 정보
    # --------------------------------------------------------

    print_information(
        data=data,
        lines=lines,
        line_count=line_count,
        path_in=path_in,
        path_out=path_out,
        path_in_radius=path_in_radius,
        path_out_radius=path_out_radius
    )

    # --------------------------------------------------------
    # 시각화
    # --------------------------------------------------------

    visualize(
        lines=lines,
        line_count=line_count,
        path_in=path_in,
        path_out=path_out,
        path_in_radius=path_in_radius,
        path_out_radius=path_out_radius,
        output_image=output_image
    )


# ============================================================
# 23. Entry Point
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\n[INFO] 종료"
        )

    except Exception as error:

        print()
        print(
            "============================================================"
        )

        print(
            "[ERROR]"
        )

        print(
            error
        )

        print(
            "============================================================"
        )

        sys.exit(
            1
        )