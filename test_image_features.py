"""
Tests for image_features.py.

Two things are being guarded here. The first is the feature layout: the vector
must be the flattened image, then four statistics, then a 256-bin histogram, in
that order, because that is the order train_model.py built it in and a Random
Forest has no way to notice if the columns are permuted. The second is the
standardisation: features must land within a few standard deviations of the
training distribution rather than thousands, which is the measurement that
distinguishes this pipeline from the one it replaces.

Tests needing the 23 MB model artifact or the practice images skip cleanly when
they are absent, so the file still runs on a bare checkout.

Run with:  python -m pytest test_image_features.py -v
pytest is a development dependency and is deliberately not added to
requirements.txt, which is the deployed app's dependency list.
"""

from __future__ import annotations

import gzip
import io
import pickle
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_features import (
    EXPECTED_N_FEATURES,
    HIST_BINS,
    TARGET_SIZE,
    ImageDecodeError,
    bgr_to_gray,
    decode_image,
    extract_features,
    features_for_image,
    preprocess_like_training,
    resize_bilinear,
    standardisation_report,
)

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "artifacts" / "alzheimer_classifier.pkl.gz"
PRACTICE_DIR = HERE / "Practice_Files"

# The practice filenames encode their own ground truth.
GROUND_TRUTH = {
    "Mild_dementia.jpg": "Mild Dementia",
    "Non-demented.jpg": "Non Demented",
    "Moderate_dementia.jpg": "Moderate Dementia",
    "Very_mild_dementia.jpg": "Very mild Dementia",
}


@pytest.fixture(scope="module")
def bundle():
    if not MODEL_PATH.is_file():
        pytest.skip("classifier artifact not available")
    with gzip.open(MODEL_PATH, "rb") as handle:
        return pickle.load(handle)


@pytest.fixture(scope="module")
def classes(bundle):
    encoder = bundle.get("label_encoder")
    if encoder is not None:
        return [str(c) for c in encoder.classes_]
    return [str(c) for c in bundle["classes"]]


@pytest.fixture(scope="module")
def practice_images():
    if not PRACTICE_DIR.is_dir():
        pytest.skip("Practice_Files not available")
    images = sorted(
        p for p in PRACTICE_DIR.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not images:
        pytest.skip("no practice images found")
    return images


def synthetic_mri(seed: int = 0) -> np.ndarray:
    """A brain-like greyscale image: black border, bright centre, texture."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:248, 0:496]
    radius = np.sqrt((y - 124.0) ** 2 + ((x - 248.0) / 2.0) ** 2)
    base = np.clip(1.0 - (radius / 110.0) ** 2, 0.0, 1.0)
    img = np.clip(base * 0.8 + rng.normal(0.0, 0.05, base.shape) * base, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Feature geometry and layout
# ---------------------------------------------------------------------------
def test_feature_count_is_pixels_plus_stats_plus_histogram():
    assert EXPECTED_N_FEATURES == 224 * 224 + 4 + 256 == 50_436


def test_preprocessing_shape_and_range():
    img = preprocess_like_training(synthetic_mri())
    assert img.shape == (TARGET_SIZE[1], TARGET_SIZE[0], 1)
    assert img.dtype == np.float32
    assert 0.0 <= float(img.min())
    assert float(img.max()) <= 1.0


def test_features_are_flattened_image_then_stats_then_histogram():
    """Length alone is not enough. The order has to match training too."""
    img = preprocess_like_training(synthetic_mri())
    features = extract_features(img)
    assert features.size == EXPECTED_N_FEATURES

    n_px = 224 * 224
    assert np.allclose(features[:n_px], img.flatten(), atol=1e-6)
    assert features[n_px] == pytest.approx(float(np.mean(img)), abs=1e-5)
    assert features[n_px + 1] == pytest.approx(float(np.std(img)), abs=1e-5)
    assert features[n_px + 2] == pytest.approx(float(np.min(img)), abs=1e-5)
    assert features[n_px + 3] == pytest.approx(float(np.max(img)), abs=1e-5)

    histogram = features[n_px + 4:]
    assert histogram.size == HIST_BINS
    assert float(histogram.sum()) == pytest.approx(1.0, abs=1e-4)


def test_features_carry_per_pixel_information():
    """
    The pipeline this replaces produced 44 distinct values tiled to length.
    A vector that is periodic with period 44 is the signature of that bug, so
    it is checked for directly.
    """
    features = features_for_image(synthetic_mri())
    assert np.unique(features).size > 200

    # Sample the middle of the image, not a corner: the first rows of an MRI
    # are uniformly black and legitimately constant.
    mid = 224 * 112
    assert not np.allclose(features[mid:mid + 44], features[mid + 44:mid + 88])
    assert float(np.std(features[mid:mid + 224])) > 0.01


# ---------------------------------------------------------------------------
# The OpenCV reproductions
# ---------------------------------------------------------------------------
def test_resize_matches_opencv():
    """
    The model was fitted on cv2.resize output. This runs only where OpenCV is
    installed, which is not the deployed app, but it is the check that earns
    the right to ship without the dependency.
    """
    cv2 = pytest.importorskip("cv2", reason="OpenCV not installed")
    source = np.repeat(synthetic_mri()[:, :, None], 3, axis=2)

    expected = cv2.resize(source, TARGET_SIZE)
    actual = resize_bilinear(source, TARGET_SIZE)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # OpenCV uses fixed-point arithmetic internally and rounds to 1/2048;
    # this uses float throughout, so single-step differences are expected.
    difference = np.abs(actual.astype(int) - expected.astype(int))
    assert difference.max() <= 1


def test_greyscale_matches_opencv():
    cv2 = pytest.importorskip("cv2", reason="OpenCV not installed")
    rng = np.random.default_rng(1)
    bgr = rng.random((32, 32, 3), dtype=np.float32)
    expected = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    assert np.allclose(bgr_to_gray(bgr), expected, atol=1e-6)


def test_module_does_not_import_opencv():
    """
    requirements.txt has no OpenCV and the deployed app's import of it falls
    through to a PIL branch, so a cv2 import here would break in production
    while passing locally.
    """
    source = (HERE / "image_features.py").read_text(encoding="utf-8")
    code_lines = [
        line for line in source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    assert not any("cv2" in line for line in code_lines), (
        "image_features.py must not import cv2 at module level"
    )


def test_histogram_excludes_pixels_at_exactly_one():
    """
    cv2.calcHist treats its upper bound as exclusive and numpy does not. On a
    bright image that difference shifts every bin once the histogram is
    normalised, so the OpenCV behaviour is reproduced.
    """
    saturated = np.ones((224, 224, 1), dtype=np.float32)
    saturated[:112] = 0.5
    features = extract_features(saturated)
    histogram = features[224 * 224 + 4:]
    assert histogram[-1] == pytest.approx(0.0)
    assert histogram[HIST_BINS // 2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def test_decode_returns_bgr_not_rgb():
    """
    Channel order is the kind of error that shows up as a small bias on every
    pixel rather than a visible failure, so it is asserted explicitly.
    """
    red_rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    red_rgb[:, :, 0] = 255
    buffer = io.BytesIO()
    Image.fromarray(red_rgb).save(buffer, format="PNG")

    decoded = decode_image(buffer.getvalue())
    assert decoded[0, 0, 2] == 255, "red belongs in the last channel for BGR"
    assert decoded[0, 0, 0] == 0


def test_decode_accepts_the_shapes_the_app_holds():
    raw = synthetic_mri()
    buffer = io.BytesIO()
    Image.fromarray(raw).save(buffer, format="PNG")
    data = buffer.getvalue()

    from_bytes = decode_image(data)
    from_handle = decode_image(io.BytesIO(data))
    from_pil = decode_image(Image.open(io.BytesIO(data)))

    assert from_bytes.shape == from_handle.shape == from_pil.shape
    assert np.array_equal(from_bytes, from_handle)
    assert np.array_equal(from_bytes, from_pil)


def test_decode_rejects_junk_clearly():
    with pytest.raises(ImageDecodeError):
        decode_image(b"this is not an image")
    with pytest.raises(ImageDecodeError):
        decode_image(Path("no_such_file_anywhere.jpg"))


def test_no_contrast_stretch_is_applied():
    """
    The app stretches the 2nd-to-98th percentile before resizing. Training
    never did, so applying it would move every pixel off the distribution the
    model learned. A low-contrast image must stay low-contrast.
    """
    dim = np.full((248, 496), 100, dtype=np.uint8)
    dim[60:180, 120:380] = 140

    processed = preprocess_like_training(dim)
    assert float(processed.max()) < 0.6, "values were stretched towards 1.0"
    assert float(processed.min()) > 0.3, "values were stretched towards 0.0"


# ---------------------------------------------------------------------------
# Behaviour against the real model
# ---------------------------------------------------------------------------
def test_features_land_inside_the_training_distribution(bundle, practice_images):
    """
    The decisive measurement. The pipeline this replaces reached max |z| of
    about 86,000 on these same images, which is what reduced the forest to a
    constant classifier.
    """
    for path in practice_images:
        report = standardisation_report(features_for_image(path), bundle["scaler"])
        assert report["n_features"] == EXPECTED_N_FEATURES
        assert report["median_abs_z"] < 2.0
        assert report["max_abs_z"] < 50.0
        assert report["fraction_beyond_6_sigma"] < 0.01


def test_practice_images_classify_to_their_labels(bundle, classes, practice_images):
    scaler, classifier = bundle["scaler"], bundle["classifier"]
    checked = 0
    for path in practice_images:
        expected = GROUND_TRUTH.get(path.name)
        if expected is None or expected not in classes:
            continue
        features = features_for_image(path)
        probabilities = classifier.predict_proba(scaler.transform(features.reshape(1, -1)))[0]
        predicted = classes[int(np.argmax(probabilities))]
        assert predicted == expected, f"{path.name}: expected {expected}, got {predicted}"
        checked += 1
    assert checked >= 1, "no practice image matched a class the model can predict"


def test_feature_count_matches_what_the_scaler_expects(bundle):
    expected = getattr(bundle["scaler"], "n_features_in_", None)
    if expected is None:
        pytest.skip("scaler does not report its input width")
    assert int(expected) == EXPECTED_N_FEATURES


def test_extraction_is_deterministic(practice_images):
    first = features_for_image(practice_images[0])
    second = features_for_image(practice_images[0])
    assert np.array_equal(first, second)


def test_path_and_bytes_give_identical_features(practice_images):
    path = practice_images[0]
    assert np.array_equal(
        features_for_image(path), features_for_image(path.read_bytes())
    )
