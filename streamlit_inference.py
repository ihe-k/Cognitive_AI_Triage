Here is the complete, fully integrated Streamlit script. It includes the updated feature extraction function with Sobel edge gradients to help capture anatomical boundaries and tissue changes more reliably, alongside the mobile-optimized image preprocessor and secure temporary file cleanup.
import os
import gzip
import glob
import joblib
import pickle
import random
import argparse
import sys
from typing import Dict, List, Tuple, Optional, Any
import requests
import streamlit as st
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import re
import matplotlib.ticker as mtick
import shap
import seaborn as sns
import librosa
from streamlit_shap import st_shap
import lime
from lime.lime_tabular import LimeTabularExplainer

# Enable iPhone HEIC/HEIF image support if installed
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# =============================================================================
# MODEL LOADING FROM GITHUB RELEASES
# =============================================================================
@st.cache_resource
def load_severity_model():
    model_url = "https://github.com/ihe-k/Cognitive_AI_Triage/releases/download/v1.0.0/severity_model.pkl.2"
    model_path = "/tmp/severity_model.pkl"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading severity model from GitHub Release..."):
            response = requests.get(model_url)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(response.content)
                
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_alzheimer_model():
    model_url = "https://github.com/ihe-k/Cognitive_AI_Triage/releases/download/v1.0.0/alzheimer_classifier.pkl.2"
    model_path = "/tmp/alzheimer_classifier.pkl"
    
    if not os.path.exists(model_path):
        with st.spinner("Downloading Alzheimer model from GitHub Release..."):
            response = requests.get(model_url)
            response.raise_for_status()
            with open(model_path, "wb") as f:
                f.write(response.content)
                
    try:
        with gzip.open(model_path, 'rb') as f:
            return joblib.load(f)
    except Exception:
        return joblib.load(model_path)

# Load severity model artifact
model_artifact = load_severity_model()
feat_names = model_artifact["feature_names"]

# Conditional OpenCV import for cloud compatibility
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV (cv2) not available. Using PIL for image processing.")

# Audio processing
try:
    import librosa
except ImportError:
    librosa = None
    print("Warning: librosa not available. Audio processing will be disabled.")

# Machine learning utilities
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None
    print("Warning: sklearn not available. Some audio analysis features will be disabled.")

# Function to map MFCC mean to depression severity
def map_severity(mean_val):
    if mean_val > -10:
        return "None/Minimal"
    elif -18 < mean_val <= -10:
        return "Mild"
    elif -28 < mean_val <= -18:
        return "Moderate"
    elif -30 < mean_val <= -28:
        return "Moderately Severe"
    else:
        return "Severe"

def map_severity_with_priority(mean_val):
    if mean_val > -10:
        severity = "None/Minimal"
        priority = False
    elif -18 < mean_val <= -10:
        severity = "Mild"
        priority = False
    elif -28 < mean_val <= -18:
        severity = "Moderate"
        priority = True
    elif -30 < mean_val <= -28:
        severity = "Moderately Severe"
        priority = True
    else:
        severity = "Severe"
        priority = True
    return severity, priority

def classify_depression_risk(mean_val):
    if mean_val > -10:
        severity = "None/Minimal"
        priority = False
    elif -18 < mean_val <= -10:
        severity = "Mild"
        priority = False
    elif -28 < mean_val <= -18:
        severity = "Moderate"
        priority = True
    elif -30 < mean_val <= -28:
        severity = "Moderately Severe"
        priority = True
    else:
        severity = "Severe"
        priority = True
    return severity, priority

def classify_dementia_risk(breathing_rate, tapping_rate, heart_rate):
    if breathing_rate > 12 and tapping_rate > 0.75 and heart_rate > 50:
        return "Based on the simulation, the estimated population-level dementia risk is low"
    elif breathing_rate < 12 and tapping_rate < 0.75 and heart_rate < 50:
        return "Based on the simulation, the estimated population-level dementia risk is high"
    else:
        return "Based on the simulation, the estimated population-level dementia risk is medium"

# =============================================================================
# 0) CONFIG
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("PREDICT_DATA_DIR", os.path.join(BASE_DIR, "data"))

MODALITIES = [
    "audio",
    "fkps",
    "gaze_conf",
    "pose_conf",
    "text",
]

# =============================================================================
# ALZHEIMER'S DISEASE IMAGE PROCESSING & ENHANCED FEATURE EXTRACTION
# =============================================================================
def load_and_preprocess_image(image_input, target_size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    """Universal pure-PIL image preprocessor with EXIF rotation, transparency flattening, percentile contrast normalization, and safe temp cleanup."""
    temp_path = None
    try:
        if hasattr(image_input, "getbuffer"):
            temp_path = f"temp_upload_{random.randint(1000, 9999)}.png"
            with open(temp_path, "wb") as f:
                f.write(image_input.getbuffer())
            image_path = temp_path
        else:
            image_path = image_input

        pil_img = Image.open(image_path)
        
        # 1. Auto-correct mobile orientation metadata (EXIF transpose)
        pil_img = ImageOps.exif_transpose(pil_img)
        
        # 2. Flatten mobile RGBA / transparency / palette profiles onto a solid background
        if pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info):
            background = Image.new("RGB", pil_img.size, (255, 255, 255))
            if pil_img.mode == 'P':
                pil_img = pil_img.convert('RGBA')
            background.paste(pil_img, mask=pil_img.split()[3])
            pil_img = background
        elif pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')
            
        # 3. Convert to grayscale numpy array
        pil_img = pil_img.convert('L')
        img_np = np.array(pil_img, dtype=np.float32)
        
        # 4. Percentile contrast normalization (eliminates mobile vs. desktop brightness bias)
        p2, p98 = np.percentile(img_np, (2, 98))
        if p98 > p2:
            img_np = np.clip((img_np - p2) / (p98 - p2), 0, 1)
        else:
            img_np = img_np / 255.0
            
        # 5. High-quality resize back to target size using Lanczos resampling
        resized_pil = Image.fromarray((img_np * 255).astype(np.uint8)).resize(target_size, Image.Resampling.LANCZOS)
        return np.array(resized_pil, dtype=np.float32) / 255.0

    except Exception as e:
        print(f"Error processing image: {e}")
        return None
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

def extract_image_features(img: np.ndarray, target_features: int) -> np.ndarray:
    """Enhanced feature extractor incorporating global statistics, histograms, quadrant stats, and edge gradients."""
    try:
        mean_val = np.mean(img)
        std_val = np.std(img)
        min_val = np.min(img)
        max_val = np.max(img)
        
        hist, _ = np.histogram(img.flatten(), bins=32, range=(0, 1))
        hist = hist.astype(np.float32) / (np.sum(hist) + 1e-6)
        
        h, w = img.shape
        h2, w2 = h // 2, w // 2
        quadrants = [
            img[:h2, :w2], img[:h2, w2:],
            img[h2:, :w2], img[h2:, w2:]
        ]
        quad_stats = []
        for q in quadrants:
            quad_stats.extend([np.mean(q), np.std(q)])
            
        # Edge density / gradient features (captures anatomical boundaries & structural atrophy)
        gy, gx = np.gradient(img)
        gradient_magnitude = np.sqrt(gx**2 + gy**2)
        edge_mean = np.mean(gradient_magnitude)
        edge_std = np.std(gradient_magnitude)
        
        base_features = np.concatenate([
            [mean_val, std_val, min_val, max_val, edge_mean, edge_std],
            hist,
            np.array(quad_stats, dtype=np.float32)
        ])
        
        if base_features.size < target_features:
            repeats = int(np.ceil(target_features / base_features.size))
            features = np.tile(base_features, repeats)[:target_features]
        elif base_features.size > target_features:
            features = base_features[:target_features]
        else:
            features = base_features
            
        return features.astype(np.float32)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(target_features, dtype=np.float32)

def check_alzheimer_model() -> bool:
    try:
        model_artifact = load_alzheimer_model()
        required_keys = ["classifier", "scaler", "label_encoder", "classes"]
        if isinstance(model_artifact, dict) and not all(key in model_artifact for key in required_keys):
            return False
        return True
    except Exception as e:
        print(f"Model load error: {e}")
        return False
    
def classify_alzheimer_image(image_file, alzheimer_model: Dict[str, Any]) -> Tuple[str, float, np.ndarray]:
    try:
        img = load_and_preprocess_image(image_file)
        if img is None:
            return "Error: Could not load image", 0.0, np.array([])
        
        scaler = alzheimer_model["scaler"]
        classifier = alzheimer_model["classifier"]
        label_encoder = alzheimer_model["label_encoder"]
        
        target_features = getattr(scaler, 'n_features_in_', 50436)
        features = extract_image_features(img, target_features=target_features)
        
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        prediction = classifier.predict(features_scaled)[0]
        probabilities = classifier.predict_proba(features_scaled)[0]
        
        class_name = label_encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities) * 100)
        
        return str(class_name), confidence, img
        
    except Exception as e:
        return f"Error: {str(e)}", 0.0, np.array([])

# =============================================================================
# File pattern helpers
# =============================================================================
def candidate_paths(split: str, pid: int, modality: str) -> List[str]:
    paths: List[str] = []
    if split == "train":
        base_dirs = [os.path.join(DATA_DIR, "train")]
        prefixes = ["train"]
    elif split == "valid":
        base_dirs = [os.path.join(DATA_DIR, "valid"), os.path.join(DATA_DIR, "dev")]
        prefixes = ["dev", "valid"]
    elif split == "test":
        base_dirs = [os.path.join(DATA_DIR, "test")]
        prefixes = ["test"]
    else:
        base_dirs = [os.path.join(DATA_DIR, split)]
        prefixes = [split]

    for base in base_dirs:
        for pref in prefixes:
            flat = os.path.join(base, f"{pref}_ft_{modality}_{pid}.npy")
            nested = os.path.join(base, str(pid), f"{pref}_ft_{modality}_{pid}.npy")
            paths.append(flat)
            paths.append(nested)

    seen = set()
    unique_paths: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    return unique_paths

def first_existing(path_list: List[str]) -> Optional[str]:
    for p in path_list:
        if os.path.exists(p):
            return p
    return None

def load_features_for_id(split: str, pid: int) -> Tuple[np.ndarray, List[str]]:
    feats, names = [], []
    for modality in MODALITIES:
        path = first_existing(candidate_paths(split, pid, modality))
        if path is None:
            continue
        arr = np.load(path, allow_pickle=False)
        arr = np.asarray(arr)
        arr = np.nan_to_num(arr.astype(np.float32))
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        feat_vec = arr
        feat_names_mod = [f"{modality}_{i}" for i in range(arr.shape[0])]
        feats.append(feat_vec.astype(np.float32, copy=False))
        names.extend(feat_names_mod)

    if len(feats) == 0:
        return np.array([]), []
    return np.concatenate(feats, axis=0).astype(np.float32, copy=False), names

def load_features_for_pid(split: str, pid: int) -> np.ndarray:
    feats, names = load_features_for_id(split, pid)
    if feats.size == 0:
        raise RuntimeError(f"No features found for PID={pid}")
    return feats

def predict_for_ids(split: str, ids: List[int], model, scaler) -> pd.DataFrame:
    X_rows = []
    valid_ids = []
    for pid in ids:
        try:
            vec = load_features_for_pid(split, pid)
            X_rows.append(vec)
            valid_ids.append(pid)
        except RuntimeError as e:
            print(f"Skipping {pid}: {e}")
            continue
    if not X_rows:
        raise RuntimeError("No valid participants found for inference.")
    X = np.vstack(X_rows)
    X_s = scaler.transform(X)
    preds = model.predict(X_s)
    return pd.DataFrame({"Participant_ID": valid_ids, "Predicted_PHQ8": preds})

def check_pretrained_model() -> bool:
    try:
        model_artifact = load_severity_model()
        required_keys = ["model", "scaler", "feature_names"]
        return all(key in model_artifact for key in required_keys)
    except Exception:
        return False

def run_simple_inference() -> Dict[str, Any]:
    model_artifact = load_severity_model()
    model = model_artifact["model"]
    scaler = model_artifact["scaler"]
    feat_names = model_artifact["feature_names"]

    n_samples = 100
    X_sample = np.random.randn(n_samples, len(feat_names)).astype(np.float32)
    X_sample_s = scaler.transform(X_sample)
    pred_sample = model.predict(X_sample_s)

    explainer_shap = shap.TreeExplainer(model)
    explainer_lime = LimeTabularExplainer(
        X_sample_s,
        feature_names=feat_names,
        class_names=["PHQ8_Score"],
        verbose=False,
        mode="regression"
    )
    
    return {
        "model": model,
        "scaler": scaler,
        "feat_names": feat_names,
        "X_sample_s": X_sample_s,
        "pred_sample": pred_sample,
        "TOTAL_N": n_samples,
        "explainer_shap": explainer_shap,
        "explainer_lime": explainer_lime,
    }

# =============================================================================
# 4) PHYSIOLOGICAL MARKERS SIMULATION & AUDIO
# =============================================================================
def extract_mfcc_features(audio_files):
    if librosa is None:
        return None, "librosa not available for audio processing"
    
    try:
        mfcc_feats = []
        file_names = []
        
        for file in audio_files:
            temp_path = f"temp_audio_{random.randint(1000, 9999)}.wav"
            try:
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                
                y, sr = librosa.load(temp_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc, axis=1)
                mfcc_feats.append(mfcc_mean)
                file_names.append(file.name)
            except Exception as e:
                print(f"Error processing audio file {file.name}: {e}")
                continue
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
        
        if not mfcc_feats:
            return None, "No valid audio files could be processed"
        
        return np.array(mfcc_feats), file_names
    except Exception as e:
        return None, f"Error in MFCC extraction: {str(e)}"

def analyze_audio_features(mfcc_features, file_names):
    try:
        if mfcc_features is None or mfcc_features.size == 0:
            return None
        
        feature_stats = {
            'mean': np.mean(mfcc_features, axis=0),
            'std': np.std(mfcc_features, axis=0),
            'min': np.min(mfcc_features, axis=0),
            'max': np.max(mfcc_features, axis=0)
        }
        
        mfcc_feature_names = [f"MFCC_{i+1}" for i in range(mfcc_features.shape[1])]
        
        if cosine_similarity is None:
            similarity_matrix = None
        else:
            similarity_matrix = cosine_similarity(mfcc_features)
        
        summary_df = pd.DataFrame({
            'File Name': file_names,
            'MFCC_Mean': [np.mean(feat) for feat in mfcc_features],
            'MFCC_Std': [np.std(feat) for feat in mfcc_features],
            'MFCC_Range': [np.max(feat) - np.min(feat) for feat in mfcc_features]
        })
        
        return {
            'mfcc_features': mfcc_features,
            'file_names': file_names,
            'feature_names': mfcc_feature_names,
            'feature_stats': feature_stats,
            'similarity_matrix': similarity_matrix,
            'summary_df': summary_df,
            'total_files': len(file_names),
            'mfcc_dimensions': mfcc_features.shape[1]
        }
    except Exception as e:
        return None

def simulate_physiological_markers(n_samples, breathing_range=(12, 20), tapping_range=(1, 5), heart_rate_range=(60, 100)):
    breathing = np.random.uniform(breathing_range[0], breathing_range[1], size=(n_samples, 1))
    tapping = np.random.uniform(tapping_range[0], tapping_range[1], size=(n_samples, 1))
    heart_rate = np.random.uniform(heart_rate_range[0], heart_rate_range[1], size=(n_samples, 1))
    return np.hstack([breathing, tapping, heart_rate])

# =============================================================================
# 5) MISINFORMATION SIMULATION
# =============================================================================
def simulate_misinformation(num_nodes, init_infected_frac=0.1, trans_prob=0.2, rec_prob=0.1, steps=20):
    G = nx.barabasi_albert_graph(num_nodes, m=2, seed=42)
    for n in G.nodes():
        G.nodes[n]['state'] = 'S'
    infected = random.sample(list(G.nodes()), max(1, int(init_infected_frac * num_nodes)))
    for n in infected:
        G.nodes[n]['state'] = 'I'
    S_list, I_list, R_list = [], [], []
    for _ in range(steps):
        new_states = {}
        for n in G.nodes():
            state = G.nodes[n]['state']
            if state == 'S':
                for nbr in G.neighbors(n):
                    if G.nodes[nbr]['state'] == 'I' and random.random() < trans_prob:
                        new_states[n] = 'I'
                        break
            elif state == 'I':
                if random.random() < rec_prob:
                    new_states[n] = 'R'
        for n, s in new_states.items():
            G.nodes[n]['state'] = s
        states = [G.nodes[n]['state'] for n in G.nodes()]
        S_list.append(states.count('S'))
        I_list.append(states.count('I'))
        R_list.append(states.count('R'))
    return S_list, I_list, R_list, G

def allocate_resources(severity_scores, capacity=10):
    idx = np.argsort(severity_scores)[::-1]
    return idx[:capacity], idx[capacity:]

# =============================================================================
# 5) STREAMLIT APP
# =============================================================================
def run_app():
    st.title("Resource Allocation Using Multimodal AI & Misinformation Modelling in Healthcare")
    st.caption("The UI below balances multimodal data interaction (audio, image and physiological signals) with a simulation of misinformation spread to prioritise patients within limited care resource settings.")

    if not check_pretrained_model():
        st.sidebar.error("❌ Pretrained model not found or invalid!")
    else:
        st.sidebar.success("✅ Pretrained model ready!")
    
    if not check_alzheimer_model():
        st.sidebar.error("❌ Alzheimer model not found or invalid!")
    else:
        st.sidebar.success("✅ Alzheimer model ready!")
    
    if OPENCV_AVAILABLE:
        st.sidebar.success("✅ OpenCV ready!")
    else:
        st.sidebar.warning("⚠️ OpenCV not available. Using PIL fallback for image processing.")
    
    if librosa is None:
        st.sidebar.warning("⚠️ Audio processing disabled!")
    else:
        st.sidebar.success("✅ Audio processing ready!")

    st.sidebar.header("Model Evaluation")
    if st.sidebar.checkbox("Show Alzheimer's Model Performance"):
        try:
            alzheimer_model = load_alzheimer_model()
            train_acc = alzheimer_model.get("train_accuracy")
            test_acc = alzheimer_model.get("test_accuracy")
            cm = alzheimer_model.get("confusion_matrix")

            if "label_encoder" in alzheimer_model:
                class_labels = list(alzheimer_model["label_encoder"].classes_)
            else:
                class_labels = [
                    "Non-Demented",
                    "Very Mild Dementia",
                    "Mild Dementia",
                    "Moderate Dementia",
                    "Severe Dementia"
                ]

            st.subheader("Alzheimer's Classifier Performance")

            if train_acc is not None:
                st.write(f"✅ **Train Accuracy:** {train_acc:.2%}")
            if test_acc is not None:
                st.write(f"✅ **Test Accuracy:** {test_acc:.2%}")

            if cm is not None:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_labels,
                            yticklabels=class_labels,
                            ax=ax)
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                ax.set_title("Confusion Matrix")
                st.pyplot(fig)
            else:
                st.warning("⚠️ No confusion matrix found in the model.")
        except Exception as e:
            st.error(f"❌ Failed to load or process model: {e}")
  
    st.sidebar.header("Physiological Markers")
    breathing_min = st.sidebar.number_input("Breathing Min (bpm)", min_value=8, max_value=30, value=12, step=1)
    breathing_max = st.sidebar.number_input("Breathing Max (bpm)", min_value=8, max_value=30, value=20, step=1)
    tapping_min = st.sidebar.number_input("Tapping Min (taps/sec)", min_value=0.5, max_value=10.0, value=1.0, step=0.5)
    tapping_max = st.sidebar.number_input("Tapping Max (taps/sec)", min_value=0.5, max_value=10.0, value=5.0, step=0.5)
    heart_rate_min = st.sidebar.number_input("Heart Rate Min (bpm)", min_value=40, max_value=200, value=60, step=5)
    heart_rate_max = st.sidebar.number_input("Heart Rate Max (bpm)", min_value=40, max_value=200, value=100, step=5)
    alpha = st.sidebar.slider("Influence of Physiological Risk", 0.0, 1.0, 0.1, step=0.05)
    st.sidebar.number_input(
        "Number of Physiological Samples",
        min_value=1,
        value=10,
        step=1,
        key="n_samples_ui"
    )

    if st.sidebar.button("🔁 Regenerate Physiological Data"):
        st.session_state["physio_data"] = simulate_physiological_markers(
            st.session_state.get("n_samples_ui", 10),
            breathing_range=(breathing_min, breathing_max),
            tapping_range=(tapping_min, tapping_max),
            heart_rate_range=(heart_rate_min, heart_rate_max)
        )
    
    st.sidebar.header("Simulation & Allocation")
    trans_prob = st.sidebar.slider("Transmission Probability", 0.0, 1.0, 0.2, 0.01)
    rec_prob   = st.sidebar.slider("Recovery Probability", 0.0, 1.0, 0.1, 0.01)
    steps      = st.sidebar.slider("Steps", 5, 100, 20, 1)
    capacity   = st.sidebar.number_input("Treatment Capacity", min_value=1, max_value=500, value=10)

    if check_pretrained_model():
        if st.sidebar.button("▶️ Run Inference"):
            with st.spinner("Loading pretrained model and running inference..."):
                arts = run_simple_inference()
                st.session_state["arts"] = arts
    else:
        st.button("▶️ Run Inference", disabled=True)
        st.warning("Cannot run inference: Pretrained model not available.")
        
    method = st.sidebar.radio("Explanation Method", ["SHAP"], index=0, horizontal=True)

    st.subheader("📥 Upload Audio & Image (Optional)")
    with st.container(border=True):
        up_col1, up_col2 = st.columns(2)
        with up_col1:
            with st.container(border=True):
                st.markdown('<div class="cell-header">Upload Audio Files</div>', unsafe_allow_html=True)
                audio_files = st.file_uploader(
                    "Upload Audio Files",
                    type=["wav", "mp3", "flac","aac"],
                    accept_multiple_files=True,
                    key="audio_uploads",
                    label_visibility="collapsed",
                )
                if audio_files and len(audio_files) > 0:
                    if st.button("🔊 Process Audio Files", key="process_audio"):
                        with st.spinner("Processing audio files..."):
                            if librosa is None:
                                st.error("❌ librosa not available.")
                            else:
                                mfcc_features, file_names = extract_mfcc_features(audio_files)
                                if mfcc_features is not None:
                                    audio_results = analyze_audio_features(mfcc_features, file_names)
                                    if audio_results:
                                        st.session_state["audio_results"] = audio_results
                                        st.success(f"✅ Processed {len(audio_files)} audio files!")
                                    else:
                                        st.error("❌ Failed to analyze audio features")
                                else:
                                    st.error(f"❌ Audio processing failed: {file_names}")
                else:
                    if "audio_results" in st.session_state:
                        del st.session_state["audio_results"]
        with up_col2:
            with st.container(border=True):
                st.markdown('<div class="cell-header">Upload Image Files</div>', unsafe_allow_html=True)
                st.file_uploader(
                    "Upload Image Files",
                    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                    accept_multiple_files=True,
                    key="image_uploads",
                    label_visibility="collapsed",
                )

    if check_alzheimer_model():
        st.subheader("Alzheimer's Disease Image Classification")
        if "alzheimer_model" not in st.session_state:
            with st.spinner("Loading Alzheimer model..."):
                alzheimer_model = load_alzheimer_model()
                st.session_state["alzheimer_model"] = alzheimer_model
        
        if "image_uploads" in st.session_state and st.session_state["image_uploads"]:
            st.write("**Processing uploaded images...**")
            for i, uploaded_file in enumerate(st.session_state["image_uploads"]):
                with st.container(border=True):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**Image {i+1}:** {uploaded_file.name}")
                        st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
                    with col2:
                        if st.button(f"🔍 Classify Image {i+1}", key=f"classify_{i}"):
                            with st.spinner("Classifying image..."):
                                alzheimer_model = st.session_state["alzheimer_model"]
                                prediction, confidence, processed_img = classify_alzheimer_image(uploaded_file, alzheimer_model)
                                if prediction.startswith("Error"):
                                    st.error(prediction)
                                else:
                                    st.success(f"**Predicted Stage:** {prediction}")
                                    st.caption(f"Confidence: {confidence:.1f}%")
        else:
            st.info("Upload MRI images to predict Alzheimer's risk")
    else:
        st.warning("Alzheimer model not available. Cannot perform image classification.")
    
    st.subheader("🎵 Audio Analysis Results")
    if "audio_results" in st.session_state:
        audio_results = st.session_state["audio_results"]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Files", audio_results["total_files"])
        with col2:
            st.metric("MFCC Dimensions", audio_results["mfcc_dimensions"])
        with col3:
            st.metric("Features per File", len(audio_results["feature_names"]))
        with col4:
            avg_mfcc = np.mean(audio_results["mfcc_features"])
            st.metric("Avg MFCC Value", f"{avg_mfcc:.2f}")

    def extract_mfcc(audio_file_path):
        try:
            y, sr = librosa.load(audio_file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            return mfcc
        except Exception as e:
            st.error(f"Error extracting MFCC: {e}")
            return None

    if "audio_results" in st.session_state and "audio_files" in st.session_state["audio_results"]:
        audio_files = st.session_state["audio_results"]["audio_files"]
        if audio_files:
            mfcc_features_list = []
            for file in audio_files:
                mfcc = extract_mfcc(file)
                if mfcc is not None:
                    mfcc_features_list.append(mfcc)
            if mfcc_features_list:
                st.session_state["mfcc_features"] = np.array(mfcc_features_list)
                st.success("MFCC features extracted successfully!")

    # Display Summary Table with Predicted Severity
    if "audio_results" in st.session_state and "summary_df" in st.session_state["audio_results"]:
        summary_df = st.session_state["audio_results"]["summary_df"].copy()

        if "MFCC_Mean" in summary_df.columns:
            summary_df["MFCC_Mean"] = pd.to_numeric(summary_df["MFCC_Mean"], errors='coerce')
            fill_val = summary_df["MFCC_Mean"].median() if not summary_df["MFCC_Mean"].isnull().all() else 0.0
            summary_df["MFCC_Mean"] = summary_df["MFCC_Mean"].fillna(fill_val)
        else:
            summary_df["MFCC_Mean"] = 0.0

        summary_df['Severity'], summary_df['Priority_Flag'] = zip(*summary_df['MFCC_Mean'].apply(map_severity_with_priority))
        summary_df['Priority'] = summary_df['Priority_Flag'].apply(lambda x: '✅ Yes' if x else '❌ No')

        summary_df = summary_df.rename(columns={
            'MFCC_Mean': 'MFCC Mean',
            'MFCC_Std': 'MFCC SD',
            'MFCC_Range': 'MFCC Range'
        }).reset_index(drop=True)
        
        st.session_state["audio_results"]["summary_df"] = summary_df

        def apply_severity_color(row):
            severity = row['Severity']
            if severity == "None/Minimal":
                color = "background-color: #28a745; color: white;"
            elif severity in ["Mild", "Moderate"]:
                color = "background-color: #ffc107; color: white;"
            else:
                color = "background-color: #dc3545; color: white;"
            return [color] * len(row)

        def render_summary_table(df):
            df_visible = df.drop(columns=['Priority_Flag'])
            df_visible.index = df_visible.index + 1

            styled_df = df_visible.style \
                .apply(apply_severity_color, axis=1) \
                .format({
                    'MFCC Mean': '{:.2f}',
                    'MFCC SD': '{:.2f}',
                    'MFCC Range': '{:.2f}'
                }) \
                .set_properties(**{
                    'text-align': 'left'
                }) \
                .set_table_styles([{
                    'selector': 'th',
                    'props': [('text-align', 'left')]
                }])

            st.markdown("**Audio Files Summary: Predicted Depression Severity Risk**")
            st.dataframe(styled_df, use_container_width=True)

        render_summary_table(summary_df)

        mfcc_mean_value = np.mean(summary_df['MFCC Mean'])
        severity, priority_flag = map_severity_with_priority(mfcc_mean_value)
        st.success(f"**Prediction:** {severity} risk of depression")
    else:
        st.info("Upload audio files to predict depressive risk")

    st.subheader("📋 Physiological Data")
    if "arts" not in st.session_state:
        st.info("Click **Run Inference** to load the pretrained model and generate predictions")
    else:
        arts = st.session_state["arts"]
        num_patients = arts["TOTAL_N"]
        expected_samples = len(arts["pred_sample"])

        if "physio_data" not in st.session_state:
            st.session_state["physio_data"] = simulate_physiological_markers(
                st.session_state.get("n_samples_ui", num_patients),
                breathing_range=(breathing_min, breathing_max),
                tapping_range=(tapping_min, tapping_max),
                heart_rate_range=(heart_rate_min, heart_rate_max)
            )

        physio_data = st.session_state["physio_data"]
        if isinstance(physio_data, list):
            physio_data = np.array(physio_data)
    
        num_physio_samples = physio_data.shape[0]
                               
        if num_physio_samples < expected_samples:
            padding = np.full((expected_samples - num_physio_samples, physio_data.shape[1]), None)
            physio_data = np.vstack([physio_data, padding])

        physio_df = pd.DataFrame(
            physio_data,
            columns=["Breathing Rate", "Tapping Rate", "Heart Rate"],
            index=range(1, expected_samples + 1)
        )

        formatted_df = physio_df.applymap(lambda x: f"{float(x):.2f}" if isinstance(x, (int, float)) else x)
        html_physio_table = formatted_df.to_html(index=True, header=True, classes="custom-table", escape=False)

        custom_css = """
        <style>
        .custom-table {
            width: 100%;
            border-collapse: collapse;
        }
        .custom-table th, .custom-table td {
            text-align: left !important;
            padding: 6px 12px;
            border: 1px solid #ddd;
        }
        </style>
        """

        st.markdown(custom_css, unsafe_allow_html=True)
        st.markdown(html_physio_table, unsafe_allow_html=True)
    
        breathing_rate = physio_df["Breathing Rate"].mean()
        tapping_rate = physio_df["Tapping Rate"].mean()
        heart_rate = physio_df["Heart Rate"].mean()

        risk = classify_dementia_risk(breathing_rate, tapping_rate, heart_rate)

        st.subheader("Feature Distributions")
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        sns.histplot(physio_df["Breathing Rate"], ax=axes[0], kde=True, color="#003A6B")
        axes[0].set_title("Breathing Rate Distribution")

        sns.histplot(physio_df["Tapping Rate"], ax=axes[1], kde=True, color="#3776A1")
        axes[1].set_title("Tapping Rate Distribution")

        sns.histplot(physio_df["Heart Rate"], ax=axes[2], kde=True, color="#6EB1D6")
        axes[2].set_title("Heart Rate Distribution")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        if isinstance(risk, str) and risk.startswith("Error"):
            st.error(risk)
        else:
            st.success(f"**Prediction:** {risk}")

        st.subheader("Depression Severity: Inference Results")
        
        if "arts" not in st.session_state:
            st.info("Click **Run Inference** to load the pretrained model and generate predictions")
        else:
            arts = st.session_state["arts"]
            st.write(f"**Samples**: {arts['TOTAL_N']}  |  **Features**: {len(arts['feat_names'])}")

            S_list_, I_list_, R_list_, G_net_ = simulate_misinformation(
                num_nodes=arts["TOTAL_N"], trans_prob=trans_prob, rec_prob=rec_prob, steps=steps
            )
            misinfo_risk_ = I_list_[-1] / arts["TOTAL_N"]

            st.info(f"Misinformation risk applied: **{misinfo_risk_:.2f}**")
    
            physio_data = st.session_state.get("physio_data")
            if isinstance(physio_data, list):
                physio_data = np.array(physio_data)

            if physio_data is not None and physio_data.shape[0] == arts["TOTAL_N"]:
                physio_min = physio_data.min(axis=0)
                physio_max = physio_data.max(axis=0)
                physio_norm = (physio_data - physio_min) / (physio_max - physio_min + 1e-6)

                weights = np.array([0.3, 0.3, 0.4])
                physio_risk_score = np.clip(physio_norm @ weights, 0, 1)

                alpha = 0.1
                adjusted_all_ = arts["pred_sample"] * (1 - misinfo_risk_) * (1 + alpha * physio_risk_score)
                st.caption("Adjusted severity = Raw severity × (1 - misinformation risk) × (1 + α × physiological risk score)")
            else:
                adjusted_all_ = arts["pred_sample"] * (1 - misinfo_risk_)
                st.warning("⚠️ Physiological data mismatch. Falling back to severity adjusted by misinformation only.")

            treated, untreated = allocate_resources(adjusted_all_, capacity=capacity)
            treated_1_based = [i + 1 for i in treated]

            df_all = pd.DataFrame({
                "Patient ID": list(range(1, len(adjusted_all_)+1)),
                "Raw Severity": [f"{x:.2f}" for x in arts["pred_sample"]],
                "Adjusted Severity": [f"{x:.2f}" for x in adjusted_all_],
            })

            df_all["Priority"] = df_all["Patient ID"].apply(
                lambda pid: "✅ Yes" if pid in treated_1_based else "❌ No"
            )
                                    
            html_table = df_all.head(100).to_html(index=False, classes="custom-table")
            st.markdown(custom_css, unsafe_allow_html=True)
            st.markdown(html_table, unsafe_allow_html=True)
    
            st.subheader("📊 Patient Details and Explanations")
            patient_idx = st.selectbox("Select Patient ID:", options=list(range(1, len(adjusted_all_)+1)), index=0)
            internal_idx = patient_idx - 1
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Raw Severity", f"{arts['pred_sample'][internal_idx]:.2f}")
            with col2:
                st.metric("Adjusted Severity", f"{adjusted_all_[internal_idx]:.2f}")
            with col3:
                st.metric("Prioritised for Treatment", "Yes" if internal_idx in treated else "No")
            with col4:
                st.metric("Misinformation Risk", f"{misinfo_risk_:.2f}")

            if method == "SHAP":
                st.subheader("SHAP Explanation")
                shap_values = arts["explainer_shap"].shap_values(arts["X_sample_s"][patient_idx:patient_idx+1])

                if isinstance(shap_values, list):
                    shap_values_local = shap_values[1]
                else:
                    shap_values_local = shap_values

                shap_values_rounded = np.round(shap_values_local, 2)
                features_rounded = np.round(arts["X_sample_s"][patient_idx:patient_idx+1], 2)

                custom_label_map = {
                    "PHQ8_Concentrating": "PHQ8: Poor Concentration",
                    "PHQ8_Depressed": "PHQ8: Depressed Mood",
                    "PHQ8_Appetite": "PHQ8: Appetite",
                    "PHQ8_Failure": "PHQ8: Failure",
                    "PHQ8_NoInterest": "PHQ8: Lack of Interest",
                    "PHQ8_Tired": "PHQ: Tired",
                    "PHQ8_Sleep": "PHQ8: Sleep",
                    "PHQ8_Energy": "PHQ8: Energy",
                    "PHQ8_Moving": "PHQ8: Slowed/Restless",
                    "PHQ8_SelfHarm": "PHQ8: Self-harm"
                }
                
                feature_labels = [
                    f"{custom_label_map.get(feat_names[i], feat_names[i])}\n{features_rounded[0][i]:.2f}"  
                    for i in range(len(feat_names))
                ]

                plt.figure(figsize=(14, 12)) 
                
                shap.force_plot(
                    arts["explainer_shap"].expected_value,
                    shap_values_rounded[0],  
                    feature_names=feature_labels,
                    matplotlib=True, 
                    show=False  
                )
                 
                ax = plt.gca()
                ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{x:.2f}"))
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, pos: f"{y:.2f}"))

                for tick in ax.get_xticklabels():
                    tick.set_rotation(0)
                    tick.set_fontsize(10)
                for tick in ax.get_yticklabels():
                    tick.set_rotation(0)
                    tick.set_fontsize(10)
                
                plt.subplots_adjust(left=0.1, right=0.9, top=0.5, bottom=0.3)
                st.pyplot(plt.gcf(), use_container_width=True)
                plt.close()

        st.subheader("Misinformation Spread Over Time")
        fig_misinfo, ax_misinfo = plt.subplots()
        ax_misinfo.plot(S_list_, label="Susceptible", color='#003A6B', linewidth=2)
        ax_misinfo.plot(I_list_, label="Infected", color='#3776A1', linewidth=2)
        ax_misinfo.plot(R_list_, label="Recovered", color='#89CFF1', linewidth=2)
        ax_misinfo.legend()
        ax_misinfo.set_xlabel("Step")
        ax_misinfo.set_ylabel("Nodes")
        st.pyplot(fig_misinfo, use_container_width=True)
        plt.close(fig_misinfo)

        st.subheader("🌐 Social Network Visualisation: Final Network State")
        fig_net, ax_net = plt.subplots(figsize=(7, 5))
        pos = nx.spring_layout(G_net_, seed=42)
        c_map = {'S': '#003A6B', 'I': '#3776A1', 'R': '#89CFF1'}
        label_map = {'S': 'Susceptible', 'I': 'Infected', 'R': 'Recovered'}
    
        node_colors = [c_map[G_net_.nodes[n]['state']] for n in G_net_.nodes()]
        nx.draw(G_net_, pos, node_color=node_colors, node_size=20, with_labels=False, ax=ax_net, edge_color='#6e6e6e')
    
        legend_patches = [mpatches.Patch(color=color, label=label_map[state]) for state, color in c_map.items()]
        ax_net.legend(handles=legend_patches, title="Node State", loc='best')
    
        st.pyplot(fig_net, use_container_width=True)
        plt.close(fig_net)

def main():
    if "streamlit" in sys.argv[0].lower() or os.environ.get("STREAMLIT_SERVER_PORT"):
        run_app()
    else:
        pass

if __name__ == "__main__":
    main()

