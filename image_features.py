"""
MRI feature extraction for the Alzheimer stage classifier.

This module exists because the feature vector the app was building did not
resemble the one the model was trained on. `train_model.py` builds 50,436
features per image: the 224x224 image flattened, then four summary statistics,
then a 256-bin intensity histogram. The app was building 44 numbers and tiling
them 1,147 times to reach the same length. Same shape, unrelated contents.

The consequence was not merely reduced accuracy. Standardised by the model's
own scaler, the tiled vector sat up to 86,000 standard deviations outside the
training distribution, and the forest degenerated into a constant classifier:
every image, whatever it showed, came back "Very mild Dementia". Rebuilding the
vector the way training builds it brings the worst feature back to about 6
standard deviations and the predictions back to the labels.

    from image_features import features_for_image
    features = features_for_image(uploaded_file)
    probabilities = classifier.predict_proba(scaler.transform(features.reshape(1, -1)))

Dependencies are numpy and Pillow only. `train_model.py` used OpenCV, but
OpenCV is not in requirements.txt and is not installed on the deployed app,
where the import at streamlit_inference.py falls through to a PIL branch. So
the OpenCV operations are reproduced here in numpy instead of adding the
dependency. That reproduction was verified against OpenCV on the labelled
practice images: identical JPEG decoding, resized pixels within 1/255, the same
predicted labels, and max |z| of 5.74 against OpenCV's 5.71. The small residue
is OpenCV's fixed-point arithmetic in its bilinear resize, which rounds to
1/2048 where this uses float throughout.

Every numbered step below mirrors `train_model.py`. Changing the order of any
of them changes the feature values, so they are kept in the original order even
where a different order would read better.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from PIL import Image, ImageOps

__all__ = [
    "TARGET_SIZE",
    "HIST_BINS",
    "EXPECTED_N_FEATURES",
    "ImageDecodeError",
    "decode_image",
    "resize_bilinear",
    "bgr_to_gray",
    "preprocess_like_training",
    "extract_features",
    "features_for_image",
    "standardisation_report",
]

# train_model.py: load_and_preprocess_image(..., target_size=(224, 224))
TARGET_SIZE: Tuple[int, int] = (224, 224)

# train_model.py: cv2.calcHist([img], [0], None, [256], [0, 1])
HIST_BINS = 256

# 50,176 pixels + mean/std/min/max + 256 histogram bins.
EXPECTED_N_FEATURES = TARGET_SIZE[0] * TARGET_SIZE[1] + 4 + HIST_BINS  # 50,436


class ImageDecodeError(ValueError):
    """Raised when an upload cannot be read as an image."""


# ---------------------------------------------------------------------------
# Step 1 - decode to BGR uint8
# ---------------------------------------------------------------------------
def decode_image(source: Any) -> np.ndarray:
    """
    Decode anything the app might hold to a 3-channel BGR uint8 array.

    BGR rather than RGB because `train_model.py` read images with cv2.imread,
    which returns BGR, and then applied cv2's BGR-to-grey weights. Decoding to
    RGB here and reusing those weights would swap the red and blue
    coefficients. On a greyscale MRI, where the channels are nearly equal, that
    swap produces a small error on every pixel rather than an obvious failure,
    which is the kind of mistake that survives testing. The channel order is
    therefore made explicit.

    Accepts a path, raw bytes, a file-like object such as Streamlit's
    UploadedFile, a PIL image, or an array that is already decoded.
    """
    if isinstance(source, np.ndarray):
        array = source
        if array.ndim == 2:
            return np.repeat(array[:, :, None], 3, axis=2).astype(np.uint8)
        if array.ndim == 3 and array.shape[2] == 4:
            return array[:, :, :3].astype(np.uint8)
        if array.ndim == 3 and array.shape[2] == 3:
            return array.astype(np.uint8)
        raise ImageDecodeError(f"unsupported array shape: {array.shape}")

    if isinstance(source, Image.Image):
        pil = source
    else:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise ImageDecodeError(f"image not found: {path}")
            handle: Any = path
        elif isinstance(source, (bytes, bytearray, memoryview)):
            handle = io.BytesIO(bytes(source))
        elif hasattr(source, "getvalue"):
            handle = io.BytesIO(source.getvalue())
        elif hasattr(source, "read"):
            handle = io.BytesIO(source.read())
        else:
            raise ImageDecodeError(f"unsupported image source: {type(source)!r}")
        try:
            pil = Image.open(handle)
            pil.load()
        except Exception as exc:  # noqa: BLE001 - any decoder failure is the same to us
            raise ImageDecodeError(f"could not decode image: {exc}") from exc

    # Honour the EXIF orientation flag before anything else, so a photographed
    # scan is not silently analysed sideways.
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")

    rgb = np.asarray(pil, dtype=np.uint8)
    return np.ascontiguousarray(rgb[:, :, ::-1])


# ---------------------------------------------------------------------------
# Step 2 - resize, reproducing cv2.resize(..., INTER_LINEAR)
# ---------------------------------------------------------------------------
def resize_bilinear(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Bilinear resize matching OpenCV's INTER_LINEAR, which is its default.

    Pillow's own BILINEAR is not a substitute. When downscaling, Pillow widens
    the filter to cover the source pixels it is averaging, so it antialiases;
    OpenCV samples four neighbours regardless of scale. The practice images are
    496x248 going to 224x224, a downscale, which is exactly where the two
    diverge. The model was fitted on OpenCV's output, so that is what is
    reproduced here.

    OpenCV maps the centre of each output pixel back to the source as
    `(i + 0.5) * scale - 0.5`, and clamps at the edges rather than
    extrapolating past them.
    """
    dst_w, dst_h = size
    src_h, src_w = image.shape[:2]

    def axis(n_dst: int, n_src: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scale = n_src / n_dst
        centres = (np.arange(n_dst, dtype=np.float64) + 0.5) * scale - 0.5
        low = np.floor(centres).astype(np.int64)
        frac = centres - low
        # Past the edges OpenCV repeats the edge pixel. Zeroing the fraction
        # where the sample falls outside reproduces that, since both
        # neighbours then clamp to the same pixel.
        frac = np.where(low < 0, 0.0, frac)
        low = np.clip(low, 0, n_src - 1)
        high = np.clip(low + 1, 0, n_src - 1)
        return low, high, frac

    x0, x1, fx = axis(dst_w, src_w)
    y0, y1, fy = axis(dst_h, src_h)

    source = image.astype(np.float64)
    flat = source.ndim == 2
    if flat:
        source = source[:, :, None]

    wx = fx[None, :, None]
    top = source[y0][:, x0] * (1.0 - wx) + source[y0][:, x1] * wx
    bottom = source[y1][:, x0] * (1.0 - wx) + source[y1][:, x1] * wx
    out = top * (1.0 - fy)[:, None, None] + bottom * fy[:, None, None]

    if image.dtype == np.uint8:
        out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return out[:, :, 0] if flat else out


# ---------------------------------------------------------------------------
# Step 3 - greyscale, reproducing cv2.cvtColor(..., COLOR_BGR2GRAY)
# ---------------------------------------------------------------------------
def bgr_to_gray(image: np.ndarray) -> np.ndarray:
    """ITU-R BT.601 luma, the weights OpenCV uses, applied in BGR order."""
    return (
        0.114 * image[:, :, 0]
        + 0.587 * image[:, :, 1]
        + 0.299 * image[:, :, 2]
    )


def preprocess_like_training(source: Any) -> np.ndarray:
    """
    Reproduce `train_model.py`'s `load_and_preprocess_image`.

    Returns a (224, 224, 1) float32 array in [0, 1].

    The order of the three operations is load-bearing: resize while still
    3-channel uint8, then divide by 255, then convert to grey. Converting to
    grey first would average different numbers, and scaling after the
    conversion would round differently. Either reintroduces the mismatch this
    module exists to remove, and neither looks wrong from the outside.

    Note what is deliberately absent: the 2nd-to-98th percentile contrast
    stretch the app applies before resizing. Training never did that, so
    applying it at inference would move every pixel away from the distribution
    the model learned.
    """
    bgr = decode_image(source)
    resized = resize_bilinear(bgr, TARGET_SIZE)
    scaled = resized.astype(np.float32) / 255.0
    grey = bgr_to_gray(scaled)
    return np.expand_dims(grey, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4 - the feature vector
# ---------------------------------------------------------------------------
def extract_features(image: np.ndarray) -> np.ndarray:
    """
    Reproduce `train_model.py`'s `extract_image_features`.

    Returns 50,436 float32 features: 50,176 flattened pixels, then mean, std,
    min and max, then a 256-bin histogram normalised to sum to 1.
    """
    if image is None or image.size == 0:
        raise ImageDecodeError("cannot extract features from an empty image")

    array = np.ascontiguousarray(image, dtype=np.float32)
    pixels = array.flatten()

    statistics = [
        float(np.mean(array)),
        float(np.std(array)),
        float(np.min(array)),
        float(np.max(array)),
    ]

    # OpenCV's histogram ranges are half-open, so a pixel of exactly 1.0 is
    # counted by numpy and dropped by cv2.calcHist. Masking those out keeps
    # this identical to training on a bright image, where the difference would
    # otherwise shift all 256 bins once they are normalised.
    values = array.reshape(-1)
    counts, _ = np.histogram(values[values < 1.0], bins=HIST_BINS, range=(0.0, 1.0))
    counts = counts.astype(np.float32)
    total = float(counts.sum())
    histogram = counts / total if total > 0 else counts

    features = np.concatenate([pixels, statistics, histogram]).astype(np.float32)
    if features.size != EXPECTED_N_FEATURES:
        raise ImageDecodeError(
            f"built {features.size} features, expected {EXPECTED_N_FEATURES}"
        )
    return features


def features_for_image(source: Any) -> np.ndarray:
    """Decode, preprocess and extract in one call. This is the entry point."""
    return extract_features(preprocess_like_training(source))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def standardisation_report(features: np.ndarray, scaler: Any) -> dict:
    """
    How far outside the training distribution a feature vector sits.

    This is the measurement that identified the original fault and the one
    that shows it is gone, so it is worth being able to print on demand. A
    healthy vector has a median around 0.5 and a maximum in the single digits.
    The tiled vector reached 86,000.
    """
    mean = np.asarray(scaler.mean_, dtype=np.float64)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    scale = np.where(scale == 0, 1.0, scale)
    z = np.abs((features.astype(np.float64) - mean) / scale)
    return {
        "median_abs_z": float(np.median(z)),
        "max_abs_z": float(np.max(z)),
        "fraction_beyond_6_sigma": float(np.mean(z > 6.0)),
        "n_features": int(features.size),
    }


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import gzip
    import pickle
    import sys

    model_path = Path(sys.argv[1] if len(sys.argv) > 1
                      else "artifacts/alzheimer_classifier.pkl.gz")
    image_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "Practice_Files")

    opener = gzip.open if model_path.suffix == ".gz" else open
    with opener(model_path, "rb") as handle:
        bundle = pickle.load(handle)

    scaler = bundle["scaler"]
    classifier = bundle["classifier"]
    encoder = bundle.get("label_encoder")
    classes = ([str(c) for c in encoder.classes_] if encoder is not None
               else [str(c) for c in bundle["classes"]])

    print(f"model   : {model_path}")
    print(f"classes : {classes}\n")

    images = sorted(p for p in image_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    for path in images:
        features = features_for_image(path)
        report = standardisation_report(features, scaler)
        probabilities = classifier.predict_proba(
            scaler.transform(features.reshape(1, -1))
        )[0]
        best = int(np.argmax(probabilities))
        print(f"{path.name}")
        print(f"   predicted : {classes[best]}  ({probabilities[best]:.1%})")
        print(f"   median |z|: {report['median_abs_z']:.3f}"
              f"   max |z|: {report['max_abs_z']:.2f}")
