import json
import os
import re

import cv2
import numpy as np
from scipy.signal import find_peaks
from pathlib import Path
import pymupdf
import pytesseract

# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_CMD = os.environ.get("TESSERACT_CMD")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# ============================================================
# 1. LOAD IMAGE
# ============================================================

PDF_PATH = None  # The Streamlit interface supplies the input PDF.

PNG_PATH = "ecg_page.png"

def pdf_to_png(
    pdf_path,
    png_path,
    page_number=0,
    dpi=300
):
    """
    Convert one PDF page into a PNG image.

    Parameters
    ----------
    pdf_path : str
        Path to the ECG PDF.

    png_path : str
        Path where the PNG should be saved.

    page_number : int
        Zero-based PDF page number.

    dpi : int
        Rendering resolution.

    Returns
    -------
    str
        Path to generated PNG.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    # --------------------------------------------------------
    # Open PDF
    # --------------------------------------------------------

    document = pymupdf.open(
        str(pdf_path)
    )

    page_count = len(document)

    if page_number >= page_count:
        document.close()

        raise ValueError(
            f"PDF has only {page_count} page(s). "
            f"Requested page {page_number}."
        )

    # --------------------------------------------------------
    # Get requested page
    # --------------------------------------------------------

    page = document.load_page(
        page_number
    )

    # --------------------------------------------------------
    # Convert DPI to zoom factor
    #
    # PDF's default rendering resolution is 72 DPI.
    #
    # Example:
    #
    # 300 / 72 = 4.1667
    # --------------------------------------------------------

    zoom = dpi / 72

    matrix = pymupdf.Matrix(
        zoom,
        zoom
    )

    # --------------------------------------------------------
    # Render page
    # --------------------------------------------------------

    pixmap = page.get_pixmap(
        matrix=matrix,

        colorspace=pymupdf.csRGB,

        alpha=False
    )

    # --------------------------------------------------------
    # Save PNG
    # --------------------------------------------------------

    pixmap.save(
        str(png_path)
    )

    document.close()

    print(
        f"PDF converted to PNG: {png_path}"
    )

    print(
        f"PNG size: "
        f"{pixmap.width} x {pixmap.height} pixels"
    )

    return str(png_path)


# ============================================================
# 2. ROTATE IMAGE
# ============================================================

def rotate_image(image, angle):
    """
    Rotate image by multiples of 90 degrees.

    angle:
        0   = no rotation
        90  = clockwise
        180 = upside down
        270 = counter-clockwise
    """

    if angle == 0:
        return image

    elif angle == 90:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_CLOCKWISE
        )

    elif angle == 180:
        return cv2.rotate(
            image,
            cv2.ROTATE_180
        )

    elif angle == 270:
        return cv2.rotate(
            image,
            cv2.ROTATE_90_COUNTERCLOCKWISE
        )

    else:
        raise ValueError(
            "Only 0, 90, 180 and 270 degrees are supported."
        )


# ============================================================
# 3. CREATE DARK-PIXEL MASK
# ============================================================

def create_dark_mask(image):
    """
    ECG trace and text are dark.
    The ECG grid is mostly red/pink.

    HSV is used so that we can suppress much of the
    colored grid while retaining dark ECG/text pixels.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    mask = (
        (saturation < 80) &
        (value < 160)
    )

    mask = mask.astype(np.uint8) * 255

    return mask


# ============================================================
# 4. CALCULATE ORIENTATION FEATURES
# ============================================================

def calculate_orientation_features(image):
    """
    Calculate how much horizontal and vertical structure
    exists in the image.

    In the canonical ECG orientation, the waveform rows
    are predominantly horizontal.
    """

    mask = create_dark_mask(image)

    lines = cv2.HoughLinesP(
        mask,

        rho=1,

        theta=np.pi / 180,

        threshold=50,

        minLineLength=100,

        maxLineGap=15
    )

    horizontal_length = 0
    vertical_length = 0

    if lines is not None:

        # Depending on the OpenCV version, lines has shape
        # (N, 1, 4) or (N, 4). Reshape to handle both.
        for x1, y1, x2, y2 in lines.reshape(-1, 4):

            angle = abs(
                np.degrees(
                    np.arctan2(
                        y2 - y1,
                        x2 - x1
                    )
                )
            ) % 180

            length = np.hypot(
                x2 - x1,
                y2 - y1
            )

            # Horizontal line
            if angle < 10 or angle > 170:

                horizontal_length += length

            # Vertical line
            elif 80 < angle < 100:

                vertical_length += length

    # --------------------------------------------------------
    # Top/bottom dark-pixel density
    #
    # In this ECG format, patient information is above
    # the waveform region.
    # --------------------------------------------------------

    height = mask.shape[0]

    top_region = mask[
        :int(0.20 * height)
    ]

    bottom_region = mask[
        int(0.80 * height):
    ]

    top_density = top_region.sum()

    bottom_density = bottom_region.sum()

    return {
        "horizontal": horizontal_length,

        "vertical": vertical_length,

        "horizontal_score":
            horizontal_length /
            (vertical_length + 1),

        "top_density":
            int(top_density),

        "bottom_density":
            int(bottom_density)
    }


# ============================================================
# 5. AUTOMATIC ORIENTATION DETECTION
# ============================================================

def detect_orientation(image):

    results = {}

    angles = [
        0,
        90,
        180,
        270
    ]

    # --------------------------------------------------------
    # Test all four possible orientations
    # --------------------------------------------------------

    for angle in angles:

        rotated = rotate_image(
            image,
            angle
        )

        features = calculate_orientation_features(
            rotated
        )

        results[angle] = features

    # --------------------------------------------------------
    # Find strongest horizontal orientation
    # --------------------------------------------------------

    best_horizontal_score = max(
        features["horizontal_score"]
        for features in results.values()
    )

    # Keep orientations reasonably close to the best one.
    #
    # This prevents tiny differences from immediately
    # eliminating the correct orientation.
    candidates = [

        angle

        for angle, features
        in results.items()

        if features["horizontal_score"]
        >= 0.90 * best_horizontal_score
    ]

    # --------------------------------------------------------
    # Tie breaker
    #
    # For this ECG layout, patient information appears
    # above the waveform region.
    # --------------------------------------------------------

    best_angle = max(

        candidates,

        key=lambda angle:

        results[angle]["top_density"] /
        (results[angle]["bottom_density"] + 1)
    )

    return best_angle, results


# ============================================================
# 6. DETECT WAVEFORM ROWS
# ============================================================

def detect_paper_extent(image):
    """
    Find the left and right edges of the ECG paper.

    The paper is covered by a pink grid, so its columns
    contain saturated pixels. The scanner background
    around it is grey/white and almost unsaturated.
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    column_fraction = (
        hsv[:, :, 1] > 30
    ).mean(axis=0)

    # Smooth so isolated specks outside the paper
    # are not mistaken for the paper edge.
    column_fraction = np.convolve(
        column_fraction,
        np.ones(25) / 25,
        mode="same"
    )

    paper_columns = np.where(
        column_fraction > 0.02
    )[0]

    if len(paper_columns) == 0:
        return 0, image.shape[1]

    return (
        int(paper_columns[0]),
        int(paper_columns[-1]) + 1
    )


def detect_waveform_rows(image):

    mask = create_dark_mask(
        image
    )

    # --------------------------------------------------------
    # Horizontal projection
    #
    # For every y-coordinate, count how many dark pixels
    # occur along that horizontal line.
    # --------------------------------------------------------

    projection = (
        mask > 0
    ).sum(axis=1).astype(float)

    # --------------------------------------------------------
    # Smooth the projection
    #
    # This turns individual ECG strokes into broader
    # waveform-region peaks.
    # --------------------------------------------------------

    smoothing_kernel = (
        np.ones(17) / 17
    )

    smooth_projection = np.convolve(
        projection,
        smoothing_kernel,
        mode="same"
    )

    # --------------------------------------------------------
    # Detect peaks
    # --------------------------------------------------------

    peaks, properties = find_peaks(

        smooth_projection,

        distance=100,

        prominence=20
    )

    image_height = image.shape[0]

    # --------------------------------------------------------
    # Ignore the very top and bottom of the page.
    #
    # Those areas contain metadata/footer rather than
    # the main waveform rows.
    # --------------------------------------------------------

    candidate_peaks = [

        int(peak)

        for peak in peaks

        if (
            0.20 * image_height
            < peak
            < 0.90 * image_height
        )
    ]

    # --------------------------------------------------------
    # Select strongest four waveform rows.
    #
    # For this ECG layout these correspond to:
    #
    # Row 1: I / aVR / V1 / V4
    # Row 2: II / aVL / V2 / V5
    # Row 3: III / aVF / V3 / V6
    # Row 4: long rhythm strip
    # --------------------------------------------------------

    candidate_peaks = sorted(

        candidate_peaks,

        key=lambda y:
        smooth_projection[y],

        reverse=True
    )

    strongest_peaks = candidate_peaks[:4]

    # Put them back in top-to-bottom order.

    waveform_centers = sorted(
        strongest_peaks
    )

    # --------------------------------------------------------
    # Create bounding boxes around each row
    # --------------------------------------------------------

    boxes = []

    # Size boxes from the spacing between rows so that tall
    # QRS spikes are not clipped. Each box takes 48% of the
    # gap on either side, leaving a small margin between rows.
    if len(waveform_centers) > 1:

        row_spacing = np.median(
            np.diff(waveform_centers)
        )

        half_height = int(
            0.48 * row_spacing
        )

    else:

        half_height = 95

    paper_x1, paper_x2 = detect_paper_extent(
        image
    )

    for center_y in waveform_centers:

        x1 = paper_x1

        y1 = max(
            0,
            center_y - half_height
        )

        x2 = paper_x2

        y2 = min(
            image_height,
            center_y + half_height
        )

        boxes.append(
            (
                x1,
                y1,
                x2,
                y2
            )
        )

    return (
        waveform_centers,
        boxes,
        projection,
        smooth_projection
    )

# ============================================================
# 7. DETECT LEAD SEPARATORS
# ============================================================

# Standard 3 x 4 layout of a 12-lead ECG, plus the rhythm strip.
# Used to name leads whenever OCR cannot read the printed label.
STANDARD_LEAD_LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"]
]

RHYTHM_LEAD = "II"

VALID_LEAD_LABELS = {
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
}


def vertical_line_profile(mask):
    """
    For every column of a row mask, count the pixels that belong
    to a long vertical structure.

    A vertical closing joins the dashes of a separator line.
    A vertical opening then removes the mostly horizontal trace
    and the label text.
    """

    height = mask.shape[0]

    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(3, height // 17))
        )
    )

    vertical = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, max(3, height // 4))
        )
    )

    return (vertical > 0).sum(axis=0)


def count_dash_pieces(mask, x, half_width=3):
    """
    Count separate dark runs along a narrow vertical strip.

    Dashed separators give many runs (5-7).
    QRS spikes and the calibration pulse are solid (1-2 runs).
    """

    strip = mask[
        :,
        max(0, x - half_width):x + half_width + 1
    ]

    column = (strip.max(axis=1) > 0).astype(np.int8)

    return int(
        np.count_nonzero(np.diff(column) == 1)
    )


def find_vertical_lines(mask):
    """
    Return x positions of long vertical structures in one row,
    with their dash-piece counts.
    """

    height = mask.shape[0]

    profile = vertical_line_profile(mask)

    peaks, _ = find_peaks(
        profile,
        height=height // 4,
        distance=5
    )

    return [
        (int(x), count_dash_pieces(mask, int(x)))
        for x in peaks
    ]


def detect_trace_extent(mask, vertical_lines):
    """
    Find where the ECG trace starts and ends inside one row.

    The trace starts after the calibration pulse, which is the
    solid vertical line near the left edge of the row.
    """

    width = mask.shape[1]

    pulse_edges = [
        x
        for x, dashes in vertical_lines
        if dashes <= 2 and x < 0.08 * width
    ]

    search_from = (
        max(pulse_edges) + 5
        if pulse_edges
        else 0
    )

    columns = np.where(
        (mask[:, search_from:] > 0).any(axis=0)
    )[0] + search_from

    if len(columns) == 0:
        return 0, width

    trace_start = int(columns[0])

    # Skip the remaining part of the pulse if a gap follows it.
    gaps = np.where(np.diff(columns) > 5)[0]

    if (
        pulse_edges
        and len(gaps) > 0
        and columns[gaps[0]] < 0.08 * width
    ):
        trace_start = int(columns[gaps[0] + 1])

    trace_end = int(columns[-1]) + 1

    return trace_start, trace_end


def detect_separators_all_rows(
    image,
    row_boxes,
    seed_dashes=4,
    confirm_dashes=3,
    min_rows=2,
    max_shift=10
):
    """
    Detect the dashed lead separators in every waveform row.

    1. In each row, find vertical lines with at least
       `seed_dashes` dash pieces.
    2. Pool them across rows. An x position seen in at least
       `min_rows` rows becomes a shared separator position.
    3. Check every row at each shared position (within
       `max_shift` pixels) using the looser `confirm_dashes`
       threshold. A row with no dashed line there (for example
       the rhythm strip) does not get that separator.

    Returns
    -------
    all_separators : list of lists
        Row-relative x positions of separators for each row.

    trace_extents : list of tuples
        Row-relative (trace_start, trace_end) for each row.
    """

    masks = []
    seeds = []
    trace_extents = []

    for x1, y1, x2, y2 in row_boxes:

        mask = create_dark_mask(
            image[y1:y2, x1:x2]
        )

        lines = find_vertical_lines(mask)

        masks.append(mask)

        trace_extents.append(
            detect_trace_extent(mask, lines)
        )

        seeds.append([
            x
            for x, dashes in lines
            if dashes >= seed_dashes
        ])

    # --------------------------------------------------------
    # Pool seeds from all rows into shared positions
    # --------------------------------------------------------

    min_rows = min(min_rows, len(row_boxes))

    pooled = sorted(
        (x, row_index)
        for row_index, row_seeds in enumerate(seeds)
        for x in row_seeds
    )

    clusters = []

    for x, row_index in pooled:

        if clusters and x - clusters[-1][-1][0] <= max_shift:
            clusters[-1].append((x, row_index))
        else:
            clusters.append([(x, row_index)])

    shared_positions = [
        int(np.median([x for x, _ in cluster]))
        for cluster in clusters
        if len({row for _, row in cluster}) >= min_rows
    ]

    # --------------------------------------------------------
    # Confirm each shared position in every row
    # --------------------------------------------------------

    all_separators = []

    for mask in masks:

        width = mask.shape[1]

        row_separators = []

        for shared_x in shared_positions:

            window = range(
                max(0, shared_x - max_shift),
                min(width, shared_x + max_shift + 1)
            )

            # Narrow strip so an adjacent QRS spike does not
            # merge with the dashes.
            dash_counts = [
                count_dash_pieces(mask, x, half_width=1)
                for x in window
            ]

            best_count = max(dash_counts)

            if best_count >= confirm_dashes:

                # Centre of all columns with the best count,
                # i.e. the middle of the dashed line.
                best_columns = [
                    x
                    for x, count in zip(window, dash_counts)
                    if count == best_count
                ]

                row_separators.append(
                    int(np.median(best_columns))
                )

        all_separators.append(row_separators)

    return all_separators, trace_extents


# ============================================================
# 8. NAME LEADS (OCR NEAR SEPARATORS + STANDARD LAYOUT)
# ============================================================

def normalize_lead_text(text):
    """
    Map raw OCR text to a valid lead name, or None.
    """

    normalized = (
        text
        .strip()
        .replace(" ", "")
        .replace("|", "I")
    )

    for valid in VALID_LEAD_LABELS:

        if normalized.upper() == valid.upper():
            return valid

    return None


def read_lead_label(image, row_box, region_x1):
    """
    OCR the small label printed just right of a lead's start.

    Only a small patch above the baseline is read, so the
    waveform does not confuse Tesseract.
    """

    x1, y1, x2, y2 = row_box

    row_height = y2 - y1

    # Start a few pixels right of the separator so its dashes
    # are not read as "I" or "L".
    offset = 8

    patch = image[
        y1:y1 + row_height // 2,
        x1 + region_x1 + offset:
        min(x2, x1 + region_x1 + offset + int(0.2 * row_height))
    ]

    if patch.size == 0:
        return None, ""

    # Tesseract expects black text on a white background.
    text_image = 255 - create_dark_mask(patch)

    scale = 4

    text_image = cv2.resize(
        text_image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_NEAREST
    )

    text_image = cv2.copyMakeBorder(
        text_image,
        20, 20, 20, 20,
        cv2.BORDER_CONSTANT,
        value=255
    )

    config = (
        "--psm 6 "
        "-c tessedit_char_whitelist="
        "IVaRLF123456"
    )

    try:
        raw_text = pytesseract.image_to_string(
            text_image,
            config=config
        ).strip()
    except pytesseract.TesseractError:
        return None, ""

    # Tesseract may return extra lines from waveform fragments;
    # accept the first line that is a valid lead name.
    for line in raw_text.splitlines():

        label = normalize_lead_text(line)

        if label is not None:
            return label, line.strip()

    return None, " ".join(raw_text.split())


def layout_lead_name(row_index, column_index, lead_count):
    """
    Lead name expected from the standard 12-lead layout.
    """

    if (
        row_index < len(STANDARD_LEAD_LAYOUT)
        and lead_count == len(STANDARD_LEAD_LAYOUT[row_index])
    ):
        return STANDARD_LEAD_LAYOUT[row_index][column_index]

    if lead_count == 1:
        return RHYTHM_LEAD

    return f"Unknown_{row_index + 1}_{column_index + 1}"


def create_lead_regions(
    image,
    row_box,
    row_index,
    separators,
    trace_extent
):
    """
    Split one waveform row into lead regions at its separators.

    Region coordinates are absolute image coordinates.
    """

    x1, y1, x2, y2 = row_box

    trace_start, trace_end = trace_extent

    boundaries = (
        [trace_start] +
        list(separators) +
        [trace_end]
    )

    lead_count = len(boundaries) - 1

    regions = []

    for column_index in range(lead_count):

        region_x1 = boundaries[column_index]
        region_x2 = boundaries[column_index + 1]

        expected = layout_lead_name(
            row_index,
            column_index,
            lead_count
        )

        ocr_label, ocr_text = read_lead_label(
            image,
            row_box,
            region_x1
        )

        # Trust the layout; OCR only confirms or flags it.
        if ocr_label is None:
            source = "layout"
        elif ocr_label == expected:
            source = "layout+ocr"
        else:
            source = "layout (ocr mismatch)"

        regions.append({

            "label": expected,

            "label_source": source,

            "ocr_text": ocr_text,

            "x1": x1 + region_x1,

            "x2": x1 + region_x2,

            "y1": y1,

            "y2": y2
        })

    return regions


# ============================================================
# 9. CALIBRATION PULSE AND BASELINE
# ============================================================

# Used when the footer does not give paper speed / gain.
DEFAULT_PAPER_SPEED_MM_PER_S = 25.0
DEFAULT_GAIN_MM_PER_MV = 10.0


def detect_calibration_pulse(mask, trace_start):
    """
    Measure the 1 mV calibration pulse at the start of a row.

    The pulse is a rectangle drawn on the row baseline:

             ______
            |      |
        ____|      |____   <- baseline (0 mV)

    Its height is exactly 1 mV, so it gives the vertical scale
    of the row, and its foot gives the 0 mV baseline.

    Parameters
    ----------
    mask : row dark-pixel mask
    trace_start : row-relative x where the ECG trace begins

    Returns
    -------
    dict with row-relative pixel positions, or None
    """

    height = mask.shape[0]

    pulse_area = mask[:, :trace_start] > 0

    if not pulse_area.any():
        return None

    column_counts = pulse_area.sum(axis=0)

    # The rising and falling edges of the pulse are the only
    # tall vertical strokes before the trace.
    edge_columns = np.where(
        column_counts > 0.15 * height
    )[0]

    if len(edge_columns) < 2:
        return None

    left_edge = int(edge_columns[0])
    right_edge = int(edge_columns[-1])

    # Top of the pulse: the horizontal line between the edges.
    inner = pulse_area[:, left_edge + 2:right_edge - 1]

    top_rows = [
        np.where(inner[:, i])[0]
        for i in range(inner.shape[1])
    ]

    top_rows = [rows for rows in top_rows if len(rows) > 0]

    if not top_rows:
        return None

    top_y = float(np.median([
        (rows.min() + rows[rows <= rows.min() + 5].max()) / 2
        for rows in top_rows
    ]))

    # Foot of the pulse: the short horizontal line before the
    # rising edge (fall back to the bottom of the edges).
    foot_columns = [
        np.where(pulse_area[:, x])[0]
        for x in range(0, left_edge - 1)
    ]

    foot_columns = [rows for rows in foot_columns if len(rows) > 0]

    if foot_columns:
        baseline_y = float(np.median([
            (rows.max() + rows[rows >= rows.max() - 5].min()) / 2
            for rows in foot_columns
        ]))
    else:
        baseline_y = float(
            np.where(pulse_area[:, left_edge])[0].max()
        )

    pulse_height = baseline_y - top_y

    if pulse_height <= 0.1 * height:
        return None

    return {
        "baseline_y": baseline_y,
        "top_y": top_y,
        "height_px": pulse_height,
        "left_x": left_edge,
        "right_x": right_edge
    }


# ============================================================
# 10. DIGITIZE ONE LEAD
# ============================================================

def remove_label_blobs(mask, baseline_y):
    """
    Remove small blobs (lead labels, specks) that sit clearly
    above the baseline and are not part of the trace.

    Returns the cleaned mask and the bounding boxes
    (x, y, w, h) of the removed blobs.
    """

    height = mask.shape[0]

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        connectivity=8
    )

    cleaned = mask.copy()

    removed = []

    for i in range(1, count):

        x, y, w, h, area = stats[i]

        is_small = (
            w < 0.12 * height and
            h < 0.12 * height
        )

        is_above_baseline = (
            y + h < baseline_y - 0.03 * height
        )

        if is_small and is_above_baseline:
            cleaned[labels == i] = 0
            removed.append((int(x), int(y), int(w), int(h)))

    return cleaned, removed


def learn_label_box(rows, margin=3):
    """
    Learn where lead labels sit, relative to the lead start
    (x) and the baseline (y), from labels that were cleanly
    separated from the trace (e.g. aVR, V1).

    All labels on a page are printed at the same offset, so
    this box also locates labels that touch the trace
    (e.g. V4-V6, whose label touches a QRS spike).

    rows : list of (lead_starts, baseline_y, removed_blobs, height)

    Returns (x_lo, x_hi, y_lo, y_hi) relative offsets, or None.
    """

    offsets = []

    for lead_starts, baseline_y, removed, height in rows:

        for x, y, w, h in removed:

            for start in lead_starts:

                if 0 <= x - start <= 0.25 * height:

                    offsets.append((
                        x - start,
                        x + w - start,
                        y - baseline_y,
                        y + h - baseline_y
                    ))

    if not offsets:
        return None

    offsets = np.array(offsets, dtype=float)

    return (
        offsets[:, 0].min() - margin,
        offsets[:, 1].max() + margin,
        offsets[:, 2].min() - margin,
        offsets[:, 3].max() + margin
    )


def erase_label_strokes(mask, lead_starts, baseline_y, label_box):
    """
    Inside the label box of every lead, erase vertical runs of
    dark pixels that lie completely inside the box.

    Label strokes are short and stay inside the box; a QRS
    spike passing through the box extends beyond it and is
    kept.
    """

    if label_box is None:
        return mask

    x_lo, x_hi, y_lo, y_hi = label_box

    height, width = mask.shape

    top = int(max(0, baseline_y + y_lo))
    bottom = int(min(height - 1, baseline_y + y_hi))

    for start in lead_starts:

        for x in range(
            int(max(0, start + x_lo)),
            int(min(width, start + x_hi + 1))
        ):

            rows = np.where(mask[:, x] > 0)[0]

            if len(rows) == 0:
                continue

            breaks = np.where(np.diff(rows) > 1)[0]

            for run in np.split(rows, breaks + 1):

                if run[0] >= top and run[-1] <= bottom:
                    mask[run, x] = 0

    return mask


def trace_lead(mask, x1, x2, baseline_y, max_gap=15, max_jump_fraction=0.25):
    """
    Follow the ECG trace column by column.

    In every column the dark run nearest to the previous point
    is chosen, so labels, grid remnants and spikes from
    neighbouring rows are ignored.

    In steep parts (QRS) one column contains a long vertical
    run; the end farthest from the previous point is used so
    that R and S peaks are not cut off.

    A run farther than `max_jump_fraction` x row height from
    the previous point is not trace (e.g. footer graphics
    reaching into the row box); that column is left empty.

    Returns
    -------
    y_values : float array, row-relative y for every column
    filled : bool array, True where the value was interpolated
    """

    width = x2 - x1

    max_jump = max_jump_fraction * mask.shape[0]

    y_values = np.full(width, np.nan)

    previous_y = baseline_y

    for i, x in enumerate(range(x1, x2)):

        rows = np.where(mask[:, x] > 0)[0]

        if len(rows) == 0:
            continue

        # Split dark pixels into contiguous runs.
        breaks = np.where(np.diff(rows) > 1)[0]

        runs = np.split(rows, breaks + 1)

        def distance(run):
            if run[0] <= previous_y <= run[-1]:
                return 0
            return min(
                abs(run[0] - previous_y),
                abs(run[-1] - previous_y)
            )

        run = min(runs, key=distance)

        if distance(run) > max_jump:
            continue

        if len(run) <= 6:
            # Flat part of the trace: centre of the line.
            y = (run[0] + run[-1]) / 2
        else:
            # Steep part: end farthest from the previous point.
            y = (
                run[0]
                if abs(run[0] - previous_y) > abs(run[-1] - previous_y)
                else run[-1]
            )

        y_values[i] = y

        previous_y = y

    # Fill short gaps (separator dashes, scan dropouts).
    missing = np.isnan(y_values)

    if missing.all():
        return y_values, missing

    known = np.where(~missing)[0]

    y_values = np.interp(
        np.arange(width),
        known,
        y_values[known]
    )

    # Do not invent long stretches of signal.
    gap_start = None

    for i, is_missing in enumerate(list(missing) + [False]):

        if is_missing and gap_start is None:
            gap_start = i

        elif not is_missing and gap_start is not None:

            if i - gap_start > max_gap:
                y_values[gap_start:i] = np.nan

            gap_start = None

    return y_values, missing


def digitize_leads(
    image,
    waveform_boxes,
    all_separators,
    trace_extents,
    all_lead_regions,
    paper_speed=DEFAULT_PAPER_SPEED_MM_PER_S,
    gain=DEFAULT_GAIN_MM_PER_MV,
    dpi=None
):
    """
    Convert every lead region into time (s) and amplitude (mV).

    Vertical scale and baseline come from each row's
    calibration pulse (1 mV high). Horizontal scale comes from
    the same pulse: pulse height / gain gives pixels per mm,
    and pixels per mm x paper speed gives pixels per second.
    """

    rows_output = []
    leads_output = []

    # --------------------------------------------------------
    # Pass 1: calibrate every row and remove free-standing
    # labels
    # --------------------------------------------------------

    row_data = []

    for row_index, (
        box,
        separators,
        extent,
        regions
    ) in enumerate(zip(
        waveform_boxes,
        all_separators,
        trace_extents,
        all_lead_regions
    )):

        bx1, by1, bx2, by2 = box

        mask = create_dark_mask(
            image[by1:by2, bx1:bx2]
        )

        trace_start, trace_end = extent

        pulse = detect_calibration_pulse(
            mask,
            trace_start
        )

        if pulse is None:
            print(
                f"  Row {row_index + 1}: calibration pulse "
                f"not found, row skipped"
            )
            continue

        baseline_y = pulse["baseline_y"]

        px_per_mv = pulse["height_px"]

        px_per_mm = px_per_mv / gain

        px_per_s = px_per_mm * paper_speed

        rows_output.append({

            "row": row_index + 1,

            "baseline_y_px": round(by1 + baseline_y, 2),

            "pulse_top_y_px": round(by1 + pulse["top_y"], 2),

            "pulse_height_px": round(px_per_mv, 2),

            "pulse_x_px": [
                bx1 + pulse["left_x"],
                bx1 + pulse["right_x"]
            ],

            "px_per_mV": round(px_per_mv, 3),

            "px_per_mm": round(px_per_mm, 3),

            "px_per_s": round(px_per_s, 3),

            "px_per_mm_from_dpi": (
                round(dpi / 25.4, 3)
                if dpi
                else None
            )
        })

        clean_mask, removed_blobs = remove_label_blobs(
            mask,
            baseline_y
        )

        row_data.append({
            "row_index": row_index,
            "box": box,
            "separators": separators,
            "regions": regions,
            "trace_start": trace_start,
            "lead_starts": [trace_start] + list(separators),
            "baseline_y": baseline_y,
            "px_per_mv": px_per_mv,
            "px_per_s": px_per_s,
            "clean_mask": clean_mask,
            "removed_blobs": removed_blobs
        })

    # --------------------------------------------------------
    # Learn the label position from cleanly separated labels
    # --------------------------------------------------------

    label_box = learn_label_box([
        (
            row["lead_starts"],
            row["baseline_y"],
            row["removed_blobs"],
            row["clean_mask"].shape[0]
        )
        for row in row_data
    ])

    # --------------------------------------------------------
    # Pass 2: erase labels touching the trace, then trace
    # --------------------------------------------------------

    for row in row_data:

        row_index = row["row_index"]
        bx1, by1, bx2, by2 = row["box"]
        separators = row["separators"]
        regions = row["regions"]
        trace_start = row["trace_start"]
        baseline_y = row["baseline_y"]
        px_per_mv = row["px_per_mv"]
        px_per_s = row["px_per_s"]

        clean_mask = erase_label_strokes(
            row["clean_mask"],
            row["lead_starts"],
            baseline_y,
            label_box
        )

        # Separator dashes would be mistaken for trace; blank a
        # thin strip around each one and let interpolation
        # bridge the trace across it.
        for x in separators:
            clean_mask[:, max(0, x - 4):x + 5] = 0

        for column_index, region in enumerate(regions):

            x1 = region["x1"] - bx1
            x2 = region["x2"] - bx1

            y_values, interpolated = trace_lead(
                clean_mask,
                x1,
                x2,
                baseline_y
            )

            x_values = np.arange(x1, x2)

            time_s = (x_values - trace_start) / px_per_s

            amplitude_mv = (baseline_y - y_values) / px_per_mv

            valid = ~np.isnan(amplitude_mv)

            leads_output.append({

                "lead": region["label"],

                "label_source": region["label_source"],

                "row": row_index + 1,

                "column": column_index + 1,

                "start_time_s": round(float(time_s[0]), 4),

                "end_time_s": round(float(time_s[-1]), 4),

                "sampling_rate_hz": round(px_per_s, 3),

                "sample_count": int(len(x_values)),

                "interpolated_samples": int(interpolated.sum()),

                "median_amplitude_mV": (
                    round(float(np.median(amplitude_mv[valid])), 4)
                    if valid.any()
                    else None
                ),

                "time_s": [
                    round(float(t), 4)
                    for t in time_s
                ],

                "amplitude_mV": [
                    round(float(a), 4) if not np.isnan(a) else None
                    for a in amplitude_mv
                ],

                "x_px": [
                    int(bx1 + x)
                    for x in x_values
                ],

                "y_px": [
                    round(float(by1 + y), 2) if not np.isnan(y) else None
                    for y in y_values
                ]
            })

    return {

        "label_box_px": (
            [round(float(v), 1) for v in label_box]
            if label_box is not None
            else None
        ),

        "units": {
            "time": "s",
            "amplitude": "mV",
            "pixels": "canonical (rotated) image coordinates"
        },

        "paper_speed_mm_per_s": paper_speed,

        "gain_mm_per_mV": gain,

        "rows": rows_output,

        "leads": leads_output
    }


def draw_digitized_traces(image, waveform_data):
    """
    Overlay the extracted traces (green) and baselines (blue)
    so the digitization can be checked visually.
    """

    overlay = image.copy()

    for row in waveform_data["rows"]:

        y = int(round(row["baseline_y_px"]))

        cv2.line(
            overlay,
            (0, y),
            (overlay.shape[1] - 1, y),
            (255, 128, 0),
            1
        )

    for lead in waveform_data["leads"]:

        points = [
            (x, int(round(y)))
            for x, y in zip(lead["x_px"], lead["y_px"])
            if y is not None
        ]

        for start, end in zip(points, points[1:]):

            if end[0] - start[0] == 1:
                cv2.line(overlay, start, end, (0, 200, 0), 2)

    return overlay


# ============================================================
# 11. METADATA OUTSIDE THE WAVEFORM ROWS
# ============================================================

# Spanish OCR keeps accents (Años, Duración). Requires
# spa.traineddata in the Tesseract tessdata folder.
OCR_LANGUAGE = "spa"


def find_text_band(mask, start_y, step, min_gap=15):
    """
    Walk from a waveform row centre (up or down) until a band
    of `min_gap` empty pixel rows is found. That gap separates
    the waveform rows from the header/footer text.
    """

    profile = (mask > 0).sum(axis=1)

    y = start_y

    empty_run = 0

    while 0 <= y < len(profile):

        empty_run = empty_run + 1 if profile[y] == 0 else 0

        if empty_run >= min_gap:
            return y

        y += step

    return min(max(y, 0), len(profile))


def ocr_region(region):
    """
    OCR a header/footer region: grid removed, black text on
    white, upscaled 2x.
    """

    if region.size == 0:
        return ""

    text_image = 255 - create_dark_mask(region)

    text_image = cv2.resize(
        text_image,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    return pytesseract.image_to_string(
        text_image,
        lang=OCR_LANGUAGE,
        config="--psm 6"
    ).strip()


def search(pattern, text):
    """
    Return the first match group exactly as printed (only
    repeated whitespace collapsed), or None.
    """

    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)

    if not match:
        return None

    return " ".join(match.group(1).split())


def leading_number(text):
    """
    Numeric part at the start of a printed value,
    e.g. "25 mm/s" -> 25.0. Returns None if not available.
    """

    if not text:
        return None

    match = re.match(r"\s*(\d+(?:[.,]\d+)?)", text)

    if not match:
        return None

    return float(match.group(1).replace(",", "."))


def parse_metadata(header_text, footer_text):
    """
    Extract the printed fields. Labels printed on the report
    (ID, Nombre, Edad, ...) are used as keys exactly as printed;
    all other keys are English. Values are kept exactly as
    printed (with units).

    OCR sometimes reads Q as O, so the QRS / QT patterns
    accept both; the keys are always written correctly.
    Fields that are not found are stored as None.
    """

    header = header_text
    footer = footer_text

    return {

        "patient": {

            "ID": search(
                r"^\s*I?D\s*:\s*(\d[\d\-]*)", header
            ),

            "Nombre": search(
                r"Nombre\s*:\s*([^\n]+)", header
            ),

            "Edad": search(
                r"Edad\s*:\s*(\d+\s*\S+)", header
            ),

            "Sexo": search(
                r"Sexo\s*:\s*(\S+)", header
            )
        },

        "record_date": search(
            r"(\d{2}[-/]\d{2}[-/]\d{4})", header
        ),

        "record_time": search(
            r"(\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?)", header
        ),

        "measurements": {

            "Frecuencia ventricular": search(
                r"Frecuencia\s+ventricular\s+(\d+\s*bpm)", header
            ),

            "Intervalo PR": search(
                r"Intervalo\s+PR\s+(\d+\s*ms)", header
            ),

            "Duración QRS": search(
                r"Duraci\S*n\s+[QO]RS\s+(\d+\s*ms)", header
            ),

            "Intervalo QT/QTc": search(
                r"Intervalo\s+[QO]T\s*/\s*[QO]Tc\s+(\d+\s*/\s*\d+\s*ms)",
                header
            ),

            "Ejes P/QRS/T": search(
                r"Ejes\s+P\s*/\s*[QO]RS\s*/\s*T\s*:?\s*"
                r"(-?\d+\s*/\s*-?\d+\s*/\s*-?\d+\s*(?:deg|°|º))",
                header
            ),

            # Printed as "QTc:Hodges" (QTc correction formula).
            "QTc": search(
                r"[QO]Tc\s*:\s*([A-Za-zÀ-ÿ]+)", header
            )
        },

        # Footer values carry no printed labels, so these keys
        # are in English (except BDR, which is printed).
        "settings": {

            "paper_speed": search(
                r"(\d+(?:[.,]\d+)?\s*mm\s*/\s*s)\b", footer
            ),

            "gain": search(
                r"(\d+(?:[.,]\d+)?\s*mm\s*/\s*mV)", footer
            ),

            "mains_filter": search(
                r"(\d+\s*Hz)\s*[~—\-]", footer
            ),

            "BDR": search(
                r"B[DO]R\s*(\d+\s*Hz)", footer
            )
        },

        "institution": search(
            r"(Pol\..*?)\s+\d{2}\.\d{2}\.\d{2}", footer
        ),

        "device": {

            "software_version": search(
                r"(\d{2}\.\d{2}\.\d{2}\s*/\s*V[\d.]+)", footer
            ),

            "SN": search(
                r"SN\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-]+)", footer
            )
        }
    }


def extract_metadata(image, waveform_centers):
    """
    OCR the text above the first waveform row (header) and
    below the last waveform row (footer).

    Returns
    -------
    metadata : dict in the output JSON format
    raw_text : dict with the OCR text of header and footer
    """

    mask = create_dark_mask(image)

    header_bottom = find_text_band(
        mask,
        waveform_centers[0],
        step=-1
    )

    footer_top = find_text_band(
        mask,
        waveform_centers[-1],
        step=1
    )

    header_text = ocr_region(image[:header_bottom])

    footer_text = ocr_region(image[footer_top:])

    metadata = parse_metadata(
        header_text,
        footer_text
    )

    raw_text = {
        "header": header_text,
        "footer": footer_text
    }

    return metadata, raw_text


def count_found_fields(metadata):
    """
    (found, total) number of leaf fields in the metadata.
    """

    found = 0
    total = 0

    for value in metadata.values():

        if isinstance(value, dict):
            sub_found, sub_total = count_found_fields(value)
            found += sub_found
            total += sub_total
        else:
            found += value is not None
            total += 1

    return found, total


# ============================================================
# 11b. OUTPUT JSON
# ============================================================

OUTPUT_SAMPLING_RATE_HZ = 500

LEAD_ORDER = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]


def resample_signal(time_s, amplitude_mv, rate):
    """
    Resample one digitized lead (one sample per pixel column)
    to a fixed sampling rate with linear interpolation.

    Samples that fall in an un-traced gap stay None.
    """

    time_s = np.asarray(time_s, dtype=float)

    amplitude = np.array(
        [np.nan if a is None else a for a in amplitude_mv],
        dtype=float
    )

    valid = ~np.isnan(amplitude)

    if valid.sum() < 2:
        return []

    count = int(np.floor((time_s[-1] - time_s[0]) * rate)) + 1

    new_time = time_s[0] + np.arange(count) / rate

    resampled = np.interp(
        new_time,
        time_s[valid],
        amplitude[valid]
    )

    nearest = np.clip(
        np.searchsorted(time_s, new_time),
        0,
        len(time_s) - 1
    )

    resampled[~valid[nearest]] = np.nan

    return [
        None if np.isnan(a) else round(float(a), 4)
        for a in resampled
    ]


def build_output_json(
    metadata,
    waveform_data,
    source,
    sampling_rate=OUTPUT_SAMPLING_RATE_HZ
):
    """
    Assemble the final JSON:

        metadata      - printed header/footer fields
        leads         - the 12 leads, resampled to `sampling_rate`
        rhythm_strip  - the long lead at the bottom
        digitization  - source file and calibration, for tracing
                        results back to the image
    """

    leads_per_row = {}

    for lead in waveform_data["leads"]:
        leads_per_row[lead["row"]] = leads_per_row.get(lead["row"], 0) + 1

    leads = {}
    rhythm_strip = None

    for lead in waveform_data["leads"]:

        entry = {

            "sampling_rate_hz": sampling_rate,

            "start_time_s": round(lead["start_time_s"], 3),

            "duration_s": round(
                lead["end_time_s"] - lead["start_time_s"], 3
            ),

            "samples_mV": resample_signal(
                lead["time_s"],
                lead["amplitude_mV"],
                sampling_rate
            )
        }

        # A row holding a single lead is the rhythm strip.
        if leads_per_row[lead["row"]] == 1:

            if rhythm_strip is None:
                rhythm_strip = {"lead": lead["lead"], **entry}

        elif lead["lead"] not in leads:
            leads[lead["lead"]] = entry

    ordered_leads = {
        name: leads[name]
        for name in LEAD_ORDER
        if name in leads
    }

    for name, entry in leads.items():
        if name not in ordered_leads:
            ordered_leads[name] = entry

    return {

        "metadata": metadata,

        "leads": ordered_leads,

        "rhythm_strip": rhythm_strip,

        "digitization": {

            **source,

            "paper_speed_mm_per_s": waveform_data["paper_speed_mm_per_s"],

            "gain_mm_per_mV": waveform_data["gain_mm_per_mV"],

            "rows": [
                {
                    "row": row["row"],
                    "baseline_y_px": row["baseline_y_px"],
                    "px_per_mV": row["px_per_mV"],
                    "px_per_s": row["px_per_s"]
                }
                for row in waveform_data["rows"]
            ],

            "lead_sources": {
                lead["lead"] if leads_per_row[lead["row"]] > 1
                else f"rhythm_{lead['lead']}":
                    lead["label_source"]
                for lead in waveform_data["leads"]
            }
        }
    }


# ============================================================
# 12. DRAW SEPARATOR DIAGNOSTIC IMAGE
# ============================================================

def draw_lead_separators(
    image,
    waveform_boxes,
    all_separators
):
    """
    Draw detected separator positions so we can visually
    inspect whether the algorithm is working.
    """

    diagnostic = image.copy()

    for row_index, (box, separators) in enumerate(
        zip(waveform_boxes, all_separators),
        start=1
    ):

        x1, y1, x2, y2 = box

        # Draw the row boundary.

        cv2.rectangle(
            diagnostic,

            (x1, y1),

            (x2 - 1, y2 - 1),

            (0, 0, 255),

            2
        )

        # Draw every detected separator.

        for separator_x in separators:

            absolute_x = (
                x1 + separator_x
            )

            cv2.line(

                diagnostic,

                (absolute_x, y1),

                (absolute_x, y2),

                (255, 0, 0),

                3
            )

            # Label it.

            cv2.putText(

                diagnostic,

                f"S{separator_x}",

                (
                    absolute_x + 5,
                    y1 + 30
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 0, 0),

                2
            )

    return diagnostic

# ============================================================
# 13. MAIN PIPELINE
# ============================================================

def process_ecg(image, dpi=None):

    print()
    print("=" * 60)
    print("ECG IMAGE ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    print()
    print("1. Detecting orientation...")

    orientation, orientation_results = (
        detect_orientation(image)
    )

    print(
        f"Detected orientation correction: "
        f"{orientation} degrees"
    )

    print()
    print("Orientation scores:")

    for angle, result in orientation_results.items():

        print(
            f"  {angle:3d} deg -> "
            f"horizontal score = "
            f"{result['horizontal_score']:.2f}"
        )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    print()
    print("2. Rotating into canonical orientation...")

    canonical_image = rotate_image(
        image,
        orientation
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    print()
    print("3. Detecting waveform rows...")

    (
        waveform_centers,
        waveform_boxes,
        projection,
        smooth_projection
    ) = detect_waveform_rows(
        canonical_image
    )

    print()
    print("Detected waveform centers:")

    for i, center in enumerate(
        waveform_centers,
        start=1
    ):

        print(
            f"  Row {i}: y = {center}"
        )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    print()
    print("4. Detected waveform boxes:")

    for i, box in enumerate(
        waveform_boxes,
        start=1
    ):

        print(
            f"  Row {i}: {box}"
        )

    # --------------------------------------------------------
    # STEP 5
    # DETECT LEAD SEPARATORS
    # --------------------------------------------------------

    print()
    print("5. Detecting lead separators...")

    all_separators, trace_extents = (
        detect_separators_all_rows(
            canonical_image,
            waveform_boxes
        )
    )

    for i, (separators, extent) in enumerate(
        zip(all_separators, trace_extents),
        start=1
    ):

        print(
            f"  Row {i}: separators = {separators}, "
            f"trace = {extent[0]} -> {extent[1]}"
        )

    # --------------------------------------------------------
    # STEP 6
    # CREATE AND NAME LEAD REGIONS
    # --------------------------------------------------------

    print()
    print("6. Creating lead regions...")

    all_lead_regions = []

    for row_index, (
        box,
        separators,
        extent
    ) in enumerate(
        zip(
            waveform_boxes,
            all_separators,
            trace_extents
        )
    ):

        regions = create_lead_regions(
            canonical_image,
            box,
            row_index,
            separators,
            extent
        )

        all_lead_regions.append(
            regions
        )

        print()
        print(f"  Row {row_index + 1}:")

        for region in regions:

            ocr_note = (
                f", ocr='{region['ocr_text']}'"
                if region["ocr_text"]
                else ""
            )

            print(
                f"    {region['label']:>3}: "
                f"x={region['x1']} -> {region['x2']}  "
                f"[{region['label_source']}{ocr_note}]"
            )

    # --------------------------------------------------------
    # STEP 7
    # METADATA (HEADER AND FOOTER TEXT)
    # --------------------------------------------------------

    print()
    print("7. Extracting metadata...")

    metadata, metadata_raw_text = extract_metadata(
        canonical_image,
        waveform_centers
    )

    found, total = count_found_fields(metadata)

    print(
        f"  {found}/{total} fields found"
    )

    if found < total:
        print(
            "  Raw header text:\n    "
            + metadata_raw_text["header"].replace("\n", "\n    ")
        )

    # --------------------------------------------------------
    # STEP 8
    # DIGITIZE LEADS (BASELINE, TIME AND AMPLITUDE)
    # --------------------------------------------------------

    print()
    print("8. Digitizing leads...")

    settings = metadata["settings"]

    paper_speed = (
        leading_number(settings["paper_speed"])
        or DEFAULT_PAPER_SPEED_MM_PER_S
    )

    gain = (
        leading_number(settings["gain"])
        or DEFAULT_GAIN_MM_PER_MV
    )

    print(
        f"  Paper speed = {paper_speed} mm/s, "
        f"gain = {gain} mm/mV"
    )

    waveform_data = digitize_leads(
        canonical_image,
        waveform_boxes,
        all_separators,
        trace_extents,
        all_lead_regions,
        paper_speed=paper_speed,
        gain=gain,
        dpi=dpi
    )

    for row in waveform_data["rows"]:

        print(
            f"  Row {row['row']}: baseline y = "
            f"{row['baseline_y_px']}, "
            f"{row['px_per_mV']} px/mV, "
            f"{row['px_per_s']} px/s"
        )

    for lead in waveform_data["leads"]:

        print(
            f"    {lead['lead']:>3}: "
            f"{lead['start_time_s']:.2f} -> "
            f"{lead['end_time_s']:.2f} s, "
            f"{lead['sample_count']} samples"
        )

    return {

        "orientation":
            orientation,

        "canonical_image":
            canonical_image,

        "waveform_centers":
            waveform_centers,

        "waveform_boxes":
            waveform_boxes,

        "orientation_results":
            orientation_results,

        "lead_separators":
            all_separators,

        "trace_extents":
            trace_extents,

        "lead_regions":
            all_lead_regions,

        "metadata":
            metadata,

        "metadata_raw_text":
            metadata_raw_text,

        "waveform_data":
            waveform_data
    }





# ============================================================
# 14. DRAW DIAGNOSTIC IMAGE
# ============================================================

def create_diagnostic_image(
    canonical_image,
    waveform_boxes
):

    diagnostic = canonical_image.copy()

    for i, box in enumerate(
        waveform_boxes,
        start=1
    ):

        x1, y1, x2, y2 = box

        # Draw bounding box

        cv2.rectangle(

            diagnostic,

            (x1, y1),

            (x2 - 1, y2 - 1),

            (0, 0, 255),

            3
        )

        # Label the detected row

        cv2.putText(

            diagnostic,

            f"ROW {i}",

            (30, y1 + 35),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.0,

            (0, 0, 255),

            2
        )

    return diagnostic

# ============================================================
# 15. DRAW LEAD REGIONS
# ============================================================

def draw_lead_regions(
    image,
    all_lead_regions
):

    diagnostic = image.copy()

    for row_regions in all_lead_regions:

        for region in row_regions:

            x1 = region["x1"]
            x2 = region["x2"]
            y1 = region["y1"]
            y2 = region["y2"]

            # Draw waveform region

            cv2.rectangle(

                diagnostic,

                (x1, y1),

                (x2, y2),

                (255, 0, 0),

                2
            )

            # Draw label position

            label_x = (
                x1 +
                10
            )

            label_y = (
                y1 +
                30
            )

            cv2.putText(

                diagnostic,

                region["label"],

                (
                    label_x,
                    label_y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (255, 0, 0),

                2
            )

    return diagnostic


# ============================================================
# 16. PROCESS ONE PDF (used by the CLI and the Streamlit app)
# ============================================================

def process_pdf(
    pdf_path,
    page_number=0,
    dpi=300,
    output_dir=None,
    png_path=None
):
    """
    Run the whole pipeline on one PDF page.

    Parameters
    ----------
    pdf_path : str or Path
    page_number : zero-based page to process
    dpi : rendering resolution
    output_dir : if given, the JSON and diagnostic images are
        saved there; otherwise nothing but the rendered PNG is
        written to disk
    png_path : where to save the rendered page
        (default: ecg_page.png in output_dir or the current folder)

    Returns
    -------
    dict with:
        output  - the final JSON data (metadata, leads,
                  rhythm_strip, digitization)
        result  - full intermediate results of process_ecg
        images  - diagnostic images (BGR numpy arrays)
    """

    pdf_path = Path(pdf_path)

    if png_path is None:
        png_path = Path(output_dir or ".") / PNG_PATH

    pdf_to_png(
        pdf_path,
        png_path,
        page_number=page_number,
        dpi=dpi
    )

    image = cv2.imread(
        str(png_path)
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load generated PNG: {png_path}"
        )

    print(
        f"Loaded ECG image: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    result = process_ecg(
        image,
        dpi=dpi
    )

    source = {
        "pdf": pdf_path.name,
        "page": page_number,
        "dpi": dpi,
        "orientation_correction_deg": result["orientation"]
    }

    output = build_output_json(
        result["metadata"],
        result["waveform_data"],
        source
    )

    images = {

        "ecg_lead_regions.png": draw_lead_regions(
            result["canonical_image"],
            result["lead_regions"]
        ),

        "ecg_lead_separators.png": draw_lead_separators(
            result["canonical_image"],
            result["waveform_boxes"],
            result["lead_separators"]
        ),

        "ecg_digitized_overlay.png": draw_digitized_traces(
            result["canonical_image"],
            result["waveform_data"]
        )
    }

    if output_dir is not None:

        output_dir = Path(output_dir)

        json_path = output_dir / "ecg_data.json"

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(
                output,
                file,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"ECG data saved to: {json_path}"
        )

        for name, diagnostic in images.items():

            cv2.imwrite(
                str(output_dir / name),
                diagnostic
            )

            print(
                f"Diagnostic image saved to: {output_dir / name}"
            )

    return {
        "output": output,
        "result": result,
        "images": images
    }


# ============================================================
# 17. RUN
# ============================================================

if __name__ == "__main__":
    raise SystemExit("Run the interface with: streamlit run app.py")
