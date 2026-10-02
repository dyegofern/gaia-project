import os

import numpy as np
from PIL import Image, ImageFilter

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "data", "chess_piece_templates.npz")
TEMPLATE_SIZE = 48
PIECE_MASK_THRESHOLD = 40
EMPTY_SQUARE_MAX_FILL = 0.05
CONFIDENCE_THRESHOLD = 0.2  # IoU threshold for piece detection


def _board_bbox(img: Image.Image):
    """Bounding box of the 8x8 grid: pixels close to either of the two
    dominant colours (the light and dark square colours)."""
    arr = np.asarray(img.convert("RGB")).astype(int)
    colours, counts = np.unique(arr.reshape(-1, 3), axis=0, return_counts=True)
    top_two = colours[np.argsort(counts)[-2:]]
    near = np.zeros(arr.shape[:2], dtype=bool)
    for c in top_two:
        near |= np.abs(arr - c).sum(axis=2) < 30
    rows = np.where(near.mean(axis=1) > 0.5)[0]
    cols = np.where(near.mean(axis=0) > 0.5)[0]
    return cols[0], rows[0], cols[-1] + 1, rows[-1] + 1


def _square_mask(square: Image.Image) -> np.ndarray:
    arr = np.asarray(square.convert("RGB")).astype(int)
    pixels, counts = np.unique(arr.reshape(-1, 3), axis=0, return_counts=True)
    background = pixels[np.argmax(counts)]
    mask = np.abs(arr - background).sum(axis=2) > PIECE_MASK_THRESHOLD
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    # Opening removes thin coordinate labels drawn in the square corners.
    mask_img = mask_img.filter(ImageFilter.MinFilter(7)).filter(ImageFilter.MaxFilter(7))
    return np.asarray(mask_img) > 0


def _normalized_silhouette(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask)
    cropped = Image.fromarray((mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] * 255).astype(np.uint8))
    return np.asarray(cropped.resize((TEMPLATE_SIZE, TEMPLATE_SIZE))) > 127


def split_squares(image_path: str):
    """Return an 8x8 list (top-to-bottom, left-to-right as displayed) of
    (piece_mask, square_image) for every square of the board image."""
    img = Image.open(image_path).convert("RGB")
    left, top, right, bottom = _board_bbox(img)
    w, h = (right - left) / 8, (bottom - top) / 8
    grid = []
    for r in range(8):
        row = []
        for c in range(8):
            box = (round(left + c * w), round(top + r * h), round(left + (c + 1) * w), round(top + (r + 1) * h))
            row.append(img.crop(box))
        grid.append(row)
    return grid


def _piece_colour(square: Image.Image, mask: np.ndarray) -> str:
    gray = np.asarray(square.convert("L"))
    return "w" if (gray[mask] > 150).mean() > 0.5 else "b"


def classify_square(square: Image.Image, templates):
    mask = _square_mask(square)
    if mask.mean() < EMPTY_SQUARE_MAX_FILL:
        return None
    silhouette = _normalized_silhouette(mask)
    best_type, best_iou = None, -1.0
    for piece_type, template in templates.items():
        iou = (silhouette & template).sum() / (silhouette | template).sum()
        if iou > best_iou:
            best_type, best_iou = piece_type, iou

    # Reject low-confidence matches
    if best_iou < CONFIDENCE_THRESHOLD:
        return None

    colour = _piece_colour(square, mask)
    return best_type.upper() if colour == "w" else best_type


def load_templates():
    data = np.load(TEMPLATE_PATH)
    return {name: data[name] for name in data.files}


def _looks_flipped(pieces) -> bool:
    """Heuristic: White normally starts at the bottom of the display, so if
    White's pieces sit higher on screen than Black's, the board is rotated."""
    white_rows = [r for r, row in enumerate(pieces) for p in row if p and p.isupper()]
    black_rows = [r for r, row in enumerate(pieces) for p in row if p and p.islower()]
    if not white_rows or not black_rows:
        return False
    return sum(white_rows) / len(white_rows) < sum(black_rows) / len(black_rows)


def board_image_to_placement(image_path: str, flipped=None) -> str:
    """FEN piece-placement field for a board image. flipped=False means
    a8 is top-left (white at the bottom); flipped=True means h1 is
    top-left (board rotated 180 degrees); None auto-detects."""
    templates = load_templates()
    grid = split_squares(image_path)
    pieces = [[classify_square(sq, templates) for sq in row] for row in grid]
    if flipped is None:
        flipped = _looks_flipped(pieces)
    if flipped:
        pieces = [row[::-1] for row in pieces[::-1]]
    ranks = []
    for row in pieces:
        out, empty = "", 0
        for p in row:
            if p is None:
                empty += 1
            else:
                out += (str(empty) if empty else "") + p
                empty = 0
        ranks.append(out + (str(empty) if empty else ""))
    return "/".join(ranks)
