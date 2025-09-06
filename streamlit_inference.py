import os
import gzip
import glob
import joblib
import pickle
import random
import argparse
import sys
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import re
import matplotlib.ticker as mtick
import shap
import seaborn as sns



# Conditional OpenCV import for cloud compatibility
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV (cv2) not available. Using PIL for image processing.")

# Optional Streamlit import for GUI mode only
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None

# Audio processing
try:
    import librosa
except ImportError:
    librosa = None
    print("Warning: librosa not available. Audio processing will be disabled.")

# Explainability
import shap
from lime.lime_tabular import LimeTabularExplainer

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

def classify_depression_risk(severity):
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
    if breathing_rate < 20 or tapping_rate > 1 or heart_rate < 100:
        return "Based on the simulation, the estimated population-level dementia risk is low"
    elif breathing_rate > 20 or tapping_rate < 1 or heart_rate > 100:
        return "Based on the simulation, the estimated population-level dementia risk is high"
    else:
        return "Based on the simulation, the estimated population-level dementia risk is medium"

# =============================================================================
# 0) CONFIG
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("PREDICT_DATA_DIR", os.path.join(BASE_DIR, "data"))



# Which modalities to use (will load if file exists)
MODALITIES = [
    "audio",
    "fkps",
    "gaze_conf",
    "pose_conf",
    "text",
]



# =============================================================================
# ALZHEIMER'S DISEASE IMAGE PROCESSING
# =============================================================================
def load_and_preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Load and preprocess an image for Alzheimer's disease classification."""
    try:
        # Load image
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            if OPENCV_AVAILABLE:
                img = cv2.imread(image_path)
                if img is None:
                    # Try PIL as fallback
                    pil_img = Image.open(image_path)
                    img = np.array(pil_img)
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    elif len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                # PIL fallback
                pil_img = Image.open(image_path)
                img = np.array(pil_img)
                # Convert RGB to BGR if needed (OpenCV format)
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = img[:, :, ::-1]  # RGB to BGR
                elif len(img.shape) == 2:
                    img = np.stack([img] * 3, axis=-1)  # Grayscale to 3-channel
        else:
            # Handle .PNG files (case sensitive)
            if OPENCV_AVAILABLE:
                img = cv2.imread(image_path)
                if img is None:
                    return None
            else:
                # PIL fallback
                try:
                    pil_img = Image.open(image_path)
                    img = np.array(pil_img)
                    # Convert RGB to BGR if needed (OpenCV format)
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = img[:, :, ::-1]  # RGB to BGR
                    elif len(img.shape) == 2:
                        img = np.stack([img] * 3, axis=-1)  # Grayscale to 3-channel
                except Exception:
                    return None

        if img is None:
            return None

        # Resize
        if OPENCV_AVAILABLE:
            img = cv2.resize(img, target_size)
        else:
            # PIL resize
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
            img = np.array(pil_img)

        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0

        # Convert to grayscale if it's a 3-channel image (for MRI/PET)
        if len(img.shape) == 3:
            if OPENCV_AVAILABLE:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                # PIL grayscale conversion
                pil_img = Image.fromarray((img * 255).astype(np.uint8))
                pil_img = pil_img.convert('L')
                img = np.array(pil_img) / 255.0
            img = np.expand_dims(img, axis=-1)

        return img

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def extract_image_features(img: np.ndarray, target_features: int = 50436) -> np.ndarray:
    """Extract features from preprocessed image with padding/truncation to match target dimensions."""
    try:
        # Flatten the image
        flattened = img.flatten()
        
        # Basic statistical features
        mean_val = np.mean(img)
        std_val = np.std(img)
        min_val = np.min(img)
        max_val = np.max(img)
        
        # Histogram features (simplified)
        hist, _ = np.histogram(img.flatten(), bins=10, range=(0, 1))
        hist = hist / np.sum(hist)  # Normalize
        
        # Combine all features
        all_features = np.concatenate([
            flattened,
            [mean_val, std_val, min_val, max_val],
            hist
        ])
        
        # Pad or truncate to match expected feature dimensions
        if all_features.size < target_features:
            # Pad with zeros if we have fewer features
            padding = np.zeros(target_features - all_features.size, dtype=np.float32)
            all_features = np.concatenate([all_features, padding])
        elif all_features.size > target_features:
            # Truncate if we have more features
            all_features = all_features[:target_features]
        
        return all_features.astype(np.float32)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.array([])

def check_alzheimer_model() -> bool:
    """Check if Alzheimer model exists and is valid."""
    model_path = os.path.join("artifacts", "alzheimer_classifier.pkl.gz")
    if not os.path.exists(model_path):
        return False
    try:
        with open(model_path, "rb") as f:
            model_artifact = pickle.load(f)
        required_keys = ["classifier", "scaler", "label_encoder", "classes"]
        if not all(key in model_artifact for key in required_keys):
            return False
        
        # Check if scaler has the expected feature count
        scaler = model_artifact["scaler"]
        if hasattr(scaler, 'n_features_in_'):
            print(f"Alzheimer model expects {scaler.n_features_in_} features")
        
        return True
    except Exception:
        return False

#def load_alzheimer_model() -> Dict[str, Any]:
#    """Load the trained Alzheimer classifier model."""
 #   model_path = os.path.join("artifacts", "alzheimer_classifier.pkl.gz")
#    with open(model_path, "rb") as f:
 #       model_artifact = pickle.load(f)
  #  return model_artifact

@st.cache_resource

def load_alzheimer_model():
    with gzip.open('artifacts/alzheimer_classifier.pkl.gz', 'rb') as f:
        model = joblib.load(f)
    return model

def check_alzheimer_model():
    try:
        _ = load_alzheimer_model()
        return True
    except Exception as e:
        print(f"Model load error: {e}")
        return False
    
def classify_alzheimer_image(image_file, alzheimer_model: Dict[str, Any]) -> Tuple[str, float, np.ndarray]:
    """Classify an uploaded image using the Alzheimer model."""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_upload_{random.randint(1000, 9999)}.png"
        with open(temp_path, "wb") as f:
            f.write(image_file.getbuffer())
        
        # Load and preprocess image
        img = load_and_preprocess_image(temp_path)
        if img is None:
            os.remove(temp_path)
            return "Error: Could not load image", 0.0, np.array([])
        
        # Extract features with proper dimension matching
        target_features = alzheimer_model["scaler"].n_features_in_
        features = extract_image_features(img, target_features=target_features)
        if features.size == 0:
            os.remove(temp_path)
            return "Error: Could not extract features", 0.0, np.array([])
        
        # Scale features
        scaler = alzheimer_model["scaler"]
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Predict
        classifier = alzheimer_model["classifier"]
        prediction = classifier.predict(features_scaled)[0]
        probabilities = classifier.predict_proba(features_scaled)[0]
        
        # Get class name and confidence
        label_encoder = alzheimer_model["label_encoder"]
        class_name = label_encoder.inverse_transform([prediction])[0]
        confidence = np.max(probabilities)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return class_name, confidence, img
        
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

    # dedupe preserve order
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

# =============================================================================
# 2) FEATURE LOADING (Same approach as demo_inference.py)
# =============================================================================
def load_features_for_id(split: str, pid: int) -> Tuple[np.ndarray, List[str]]:
    """Load & align features for one participant using the same approach as demo_inference.py."""
    feats, names = [], []
    for modality in MODALITIES:
        path = first_existing(candidate_paths(split, pid, modality))
        if path is None:
            continue
        arr = np.load(path, allow_pickle=False)
        arr = np.asarray(arr)
        # Use the same summarization as demo_inference.py
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
    """Load & align features for one participant - same as demo_inference.py."""
    feats, names = load_features_for_id(split, pid)
    if feats.size == 0:
        raise RuntimeError(f"No features found for PID={pid}")
    return feats

def predict_for_ids(split: str, ids: List[int], model, scaler) -> pd.DataFrame:
    """Run inference on a list of participant IDs - same as demo_inference.py.
    
    Usage example (same as demo_inference.py):
        participant_ids = [300, 301]
        results = predict_for_ids("test", participant_ids, model, scaler)
        print(results)
        results.to_csv("artifacts/inference_results.csv", index=False)
    """
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
    """Check if pretrained model exists and is valid."""
    model_path = os.path.join("artifacts", "severity_model.pkl")
    if not os.path.exists(model_path):
        return False
    try:
        with open(model_path, "rb") as f:
            model_artifact = pickle.load(f)
        required_keys = ["model", "scaler", "feature_names"]
        return all(key in model_artifact for key in required_keys)
    except Exception:
        return False

def run_simple_inference() -> Dict[str, Any]:
    """Run simple inference using pretrained model."""
    # Load pretrained model
    model_path = os.path.join("artifacts", "severity_model.pkl")
    with open(model_path, "rb") as f:
        model_artifact = pickle.load(f)
    
    model = model_artifact["model"]
    scaler = model_artifact["scaler"]
    feat_names = model_artifact["feature_names"]

    # For demo purposes, create some sample data
    # In real usage, you would load actual test data
    n_samples = 100
    X_sample = np.random.randn(n_samples, len(feat_names)).astype(np.float32)
    X_sample_s = scaler.transform(X_sample)
    pred_sample = model.predict(X_sample_s)

####

####
    
    # Create explainability objects
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
# 4) PHYSIOLOGICAL MARKERS SIMULATION
# =============================================================================

# --- Audio MFCC features ---
def extract_mfcc_features(audio_files):
    """Extract MFCC features from audio files."""
    if librosa is None:
        return None, "librosa not available for audio processing"
    
    try:
        mfcc_feats = []
        file_names = []
        
        for file in audio_files:
            # Save uploaded file temporarily
            temp_path = f"temp_audio_{random.randint(1000, 9999)}.wav"
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            
            try:
                y, sr = librosa.load(temp_path, sr=None)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_mean = np.mean(mfcc, axis=1)
                mfcc_feats.append(mfcc_mean)
                file_names.append(file.name)
                
                # Clean up temp file
                os.remove(temp_path)
                
            except Exception as e:
                print(f"Error processing audio file {file.name}: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                continue
        
        if not mfcc_feats:
            return None, "No valid audio files could be processed"
        
        return np.array(mfcc_feats), file_names
        
    except Exception as e:
        return None, f"Error in MFCC extraction: {str(e)}"

def analyze_audio_features(mfcc_features, file_names):
    """Analyze extracted MFCC features and create results similar to inference results."""
    try:
        if mfcc_features is None or mfcc_features.size == 0:
            return None
        
        # Calculate basic statistics for each feature
        feature_stats = {
            'mean': np.mean(mfcc_features, axis=0),
            'std': np.std(mfcc_features, axis=0),
            'min': np.min(mfcc_features, axis=0),
            'max': np.max(mfcc_features, axis=0)
        }
        
        # Create feature names for MFCC coefficients
        mfcc_feature_names = [f"MFCC_{i+1}" for i in range(mfcc_features.shape[1])]
        
        # Calculate similarity matrix between audio files
        if cosine_similarity is None:
            print("Warning: sklearn not available for cosine similarity. Skipping audio similarity matrix.")
            similarity_matrix = None
        else:
            similarity_matrix = cosine_similarity(mfcc_features)
        
        # Create a summary dataframe
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
    """
    Simulate physiological markers with customizable ranges.
    Args:
        n_samples: Number of samples to generate
        breathing_range: Tuple of (min, max) breathing rate in breaths per minute
        tapping_range: Tuple of (min, max) tapping rate in taps per second  
        heart_rate_range: Tuple of (min, max) heart rate in beats per minute
    Returns:
        Array of shape (n_samples, 3) with [breathing, tapping, heart_rate]
    """
####
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
# 5) STREAMLIT APP (NEW UI)
# =============================================================================
def run_app():
    # ---- Header ----
    st.title("Resource Allocation Using Multimodal AI & Misinformation Modelling in Healthcare")
    st.caption("The UI below balances multimodal data interaction (audio, image and physiological signals) with a simulation of misinformation spread to prioritise patients within limited care resource settings.")

    # Model status check
    if not check_pretrained_model():
        st.sidebar.error("❌ Pretrained model not found or invalid!")
        st.sidebar.info("Please ensure 'artifacts/severity_model.pkl' exists and contains a valid model.")
    else:
        st.sidebar.success("✅ Pretrained model ready!")
    
    # Alzheimer model status check
    if not check_alzheimer_model():
        st.sidebar.error("❌ Alzheimer model not found or invalid!")
        st.sidebar.info("Please ensure 'artifacts/alzheimer_classifier.pkl.gz' exists and contains a valid model.")
    else:
        st.sidebar.success("✅ Alzheimer model ready!")
    
    # OpenCV status check
    if OPENCV_AVAILABLE:
        st.sidebar.success("✅ OpenCV ready!")
    else:
        st.sidebar.warning("⚠️ OpenCV not available. Using PIL fallback for image processing. Some advanced image features may be limited")
      #  st.sidebar.info("ℹ️ Note: OpenCV  is not available. Using PIL fallback for image processing. Some advanced image features may be limited.")
    
    # Audio processing status check
    if librosa is None:
        st.sidebar.warning("⚠️ Audio processing disabled!")
        st.sidebar.info("Install librosa for audio file analysis: pip install librosa")
    else:
        st.sidebar.success("✅ Audio processing ready!")

 #### 
    st.sidebar.header("Model Evaluation")
    if st.sidebar.checkbox("Show Alzheimer's Model Performance"):
        try:
            alzheimer_model = load_alzheimer_model()
            train_acc = alzheimer_model.get("train_accuracy")
            test_acc = alzheimer_model.get("test_accuracy")
            cm = alzheimer_model.get("confusion_matrix")

            # Define class names in order of index
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
#####
  
    # Physiological markers controls
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
    
    # Sidebar controls (misinfo + capacity)
    st.sidebar.header("Simulation & Allocation")
    trans_prob = st.sidebar.slider("Transmission Probability", 0.0, 1.0, 0.2, 0.01)
    rec_prob   = st.sidebar.slider("Recovery Probability", 0.0, 1.0, 0.1, 0.01)
    steps      = st.sidebar.slider("Steps", 5, 100, 20, 1)
    capacity   = st.sidebar.number_input("Treatment Capacity", min_value=1, max_value=500, value=10)

            # Run inference button
    
    if check_pretrained_model():
        if st.sidebar.button("▶️ Run Inference"):
            with st.spinner("Loading pretrained model and running inference..."):
                arts = run_simple_inference()
                st.session_state["arts"] = arts
    else:
        st.button("▶️ Run Inference", disabled=True)
        st.warning("Cannot run inference: Pretrained model not available.")
        
    method     = st.sidebar.radio("Explanation Method", ["LIME", "SHAP"], index=0, horizontal=True)
   


  

    # Uploaders (optional)
    st.subheader("📥 Upload Audio & Image (Optional)")
    st.markdown(
        """
        <style>
        /* Tighten inner cell padding and add header row vibe */
        .cell-header { font-weight: 700; border-bottom: 1px solid #bbb; margin: -0.25rem -0.25rem 0.5rem -0.25rem; padding: 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        up_col1, up_col2 = st.columns(2)
        with up_col1:
            with st.container(border=True):
                st.markdown('<div class="cell-header">Upload Audio Files</div>', unsafe_allow_html=True)
                st.markdown('<div class="upload-box">', unsafe_allow_html=True)
                audio_files = st.file_uploader(
                    "Upload Audio Files",
                    type=["wav", "mp3", "flac","aac"],
                    accept_multiple_files=True,
                    key="audio_uploads",
                    label_visibility="collapsed",
                    
                )
                # Add audio processing button
                if audio_files and len(audio_files) > 0:
                    if st.button("🔊 Process Audio Files", key="process_audio"):
                        with st.spinner("Processing audio files..."):
                            if librosa is None:
                                st.error("❌ librosa not available. Please install it for audio processing.")
                            else:
                                # Extract MFCC features
                                mfcc_features, file_names = extract_mfcc_features(audio_files)
                                
                                if mfcc_features is not None:
                                    # Analyze features
                                    audio_results = analyze_audio_features(mfcc_features, file_names)
                                    if audio_results:
                                        st.session_state["audio_results"] = audio_results
                                        st.success(f"✅ Processed {len(audio_files)} audio files!")
                                    else:
                                        st.error("❌ Failed to analyze audio features")
                                else:
                                    st.error(f"❌ Audio processing failed: {file_names}")
                else:
                    # Clear audio results when no audio files are uploaded
                    if "audio_results" in st.session_state:
                        del st.session_state["audio_results"]
        with up_col2:
            with st.container(border=True):
                st.markdown('<div class="cell-header">Upload Image Files</div>', unsafe_allow_html=True)
                st.markdown('<div class="upload-box">', unsafe_allow_html=True)
                st.file_uploader(
                    "Upload Image Files",
                    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                    accept_multiple_files=True,
                    key="image_uploads",
                    label_visibility="collapsed",
                    
                )
       # with up_col3:
          #  with st.container(border=True):
          #      st.markdown('<div class="cell-header">n Samples ▼</div>', unsafe_allow_html=True)
          #      st.number_input(
           #         "n Samples",
           #         min_value=1,
            #        value=10,
            #        step=1,
             #       key="n_samples_ui",
             #       label_visibility="collapsed",
              #  )

        #st.sidebar.header("Sample Settings")
        #st.sidebar.number_input(
        #    "Number of Physiological Samples",
        #    min_value=1,
        #    value=st.session_state.get("n_samples_ui", 10),
         #   step=1,
         #   key="n_samples_ui",
         #   label_visibility="visible",
        #)

    # Alzheimer Image Classification
    if check_alzheimer_model():
        st.subheader("Alzheimer's Disease Image Classification")

        #####


        ###

        
        # Show OpenCV status for image processing
      #  if not OPENCV_AVAILABLE:
       #     st.info("ℹ️ **Note:** OpenCV is not available. Using PIL for image processing. This may affect some advanced image features.")
        
        # Load Alzheimer model
        if "alzheimer_model" not in st.session_state:
            with st.spinner("Loading Alzheimer model..."):
                alzheimer_model = load_alzheimer_model()
                st.session_state["alzheimer_model"] = alzheimer_model
        
        # Process uploaded images
        if "image_uploads" in st.session_state and st.session_state["image_uploads"]:
            st.write("**Processing uploaded images...**")
            
            for i, uploaded_file in enumerate(st.session_state["image_uploads"]):
                with st.container(border=True):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.write(f"**Image {i+1}:** {uploaded_file.name}")
                        # Display the uploaded image
                        st.image(uploaded_file, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
                    
                    with col2:
                        if st.button(f"🔍 Classify Image {i+1}", key=f"classify_{i}"):
                            with st.spinner("Classifying image..."):
                                alzheimer_model = st.session_state["alzheimer_model"]
                                prediction, confidence, processed_img = classify_alzheimer_image(uploaded_file, alzheimer_model)
                                
                                if prediction.startswith("Error"):
                                    st.error(prediction)
                                else:
                                    # Display results - only prediction
                                    st.success(f"**Prediction:** {prediction}")
        else:
            st.info("Upload MRI images to predict Alzheimer's risk")
    else:
        st.warning("Alzheimer model not available. Cannot perform image classification.")
    
    # Audio Analysis Results
    st.subheader("🎵 Audio Analysis Results")
    
    if "audio_results" in st.session_state:
        
        audio_results = st.session_state["audio_results"]
        
        # Display basic info
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
 #### 
    def extract_mfcc(audio_file_path):
        try:
            # Load the audio file
            y, sr = librosa.load(audio_file_path, sr=None)
            # Extract MFCCs (Mel-frequency cepstral coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            return mfcc
        except Exception as e:
            st.error(f"Error extracting MFCC: {e}")
            return None

# Function to map MFCC mean to depression severity
    #def map_severity(mean_val):
    #    if mean_val > -10:
     #       return "None/Minimal"
      #  elif -18 < mean_val <= -10:
      #      return "Mild"
      #  elif -28 < mean_val <= -18:
       #     return "Moderate"
       # elif -30 < mean_val <= -28:
        #    return "Moderately Severe"
       # else:
       #     return "Severe"

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
    

    # Ensure audio file is uploaded and MFCC features are extracted
    if "audio_results" in st.session_state and "audio_files" in st.session_state["audio_results"]:
        audio_files = st.session_state["audio_results"]["audio_files"]

        if audio_files:
            # Extract MFCC features for each file
            mfcc_features_list = []
            for file in audio_files:
                mfcc = extract_mfcc(file)
                if mfcc is not None:
                    mfcc_features_list.append(mfcc)

        # Store the MFCC features in session state
            if mfcc_features_list:
                st.session_state["mfcc_features"] = np.array(mfcc_features_list)  # Store as a numpy array
                st.success("MFCC features extracted successfully!")
            else:
                st.warning("No MFCC features extracted from the audio files.")
                
    def display_dataframe(df, priority_column_name): 
        if df is None:
            st.warning("DataFrame is empty or None.")
            return
        try:
    #     Create a copy to avoid modifying the original DataFrame
            df_copy = df.copy()

        # Hide the specified column (Important: This is done *before* styling)
            if priority_column_name in df_copy.columns:
                def hide_column(s):
                    return ['display: none;' if col == priority_column_name else '' for col in s.index]
                styled_df = df_copy.style.apply(hide_column, axis=0)

                st.dataframe(styled_df)

            else:
            
                st.dataframe(df_copy)
        except Exception as e:
            st.error(f"Error displaying DataFrame: {e}")
            
            # Example with MFCC features (combining with your existing code)
            if "mfcc_features" in st.session_state:
                mfcc_features_list = st.session_state["mfcc_features"]

                # Check if mfcc_features_list is valid
                if mfcc_features_list is not None and len(mfcc_features_list) > 0:
                    # Example assuming mfcc_features_list is a list of lists (adjust as needed)
                    mfcc_df = pd.DataFrame(mfcc_features_list)
                    display_dataframe(mfcc_df, "MFCC_feature") # Replace "MFCC_feature" with your column name if needed
                else:
                    st.warning("No valid MFCC features to display.")

    # Display Summary Table with Predicted Severity
    if "audio_results" in st.session_state and "summary_df" in st.session_state["audio_results"]:
        summary_df = st.session_state["audio_results"]["summary_df"].copy()

        # Ensure 'MFCC_Mean' is numeric
        summary_df["MFCC_Mean"] = pd.to_numeric(summary_df["MFCC_Mean"], errors='coerce')  # Coerce non-numeric values to NaN
        summary_df["MFCC_Mean"].fillna(0, inplace=True)  # Replace NaNs with 0 (or another suitable value)

        # Add Predicted Severity column
        # summary_df["Predicted Severity"] = summary_df["MFCC_Mean"].apply(map_severity)

        # Map severity and priority
        summary_df['Severity'], summary_df['Priority_Flag'] = zip(*summary_df['MFCC_Mean'].apply(map_severity_with_priority))

       # summary_df['Severity'], summary_df['Priority_Flag'] = zip(*summary_df['MFCC_Mean'].apply(map_severity_with_priority))

        summary_df['Priority'] = summary_df['Priority_Flag'].apply(lambda x: '✅ Yes' if x else '❌ No')

        # Rename columns to more descriptive names
        summary_df_renamed = summary_df.rename(columns={
            'MFCC_Mean': 'MFCC Mean',
            'MFCC_Std': 'MFCC SD',
            'MFCC_Range': 'MFCC Range'
        })

        # Reset index for styling
        summary_df_reset = summary_df.drop(columns=['Priority_Flag']).reset_index(drop=True)

        # Define color styling based on severity
        def apply_severity_color(row):
            severity = row['Severity']
            if severity == "None/Minimal":
                color = "background-color: #28a745; color: white;"  # Green
            elif severity in ["Mild", "Moderate"]:
                color = "background-color: #ffc107; color: black;"  # Amber
            else:  # "Moderately Severe" or "Severe"
                color = "background-color: #dc3545; color: white;"  # Red
            return [color] * len(row)

        # Apply styling
        #styled_df = summary_df_reset.style.apply(apply_severity_color, axis=1)
        styled_df = summary_df_renamed.style.apply(apply_severity_color, axis=1)

        # Format numeric columns
        styled_df = styled_df.format({
            'MFCC Mean': '{:.2f}', 
            'MFCC SD': '{:.2f}', 
            'MFCC Range': '{:.2f}'
        })


        # Combine CSS and table HTML
        html_with_style = styled_df.to_html()

        # Reset index for styling and set the starting index to 1
        summary_df_reset = summary_df_renamed.drop(columns=['Priority_Flag']).reset_index(drop=True)
        summary_df_reset.index = summary_df_reset.index + 1  # Adjusting the index to start from 1

        # Apply styling to hide 'Priority_Flag' and display the styled DataFrame
        def hide_column(s):
            return ['display: none;' if col == 'Priority_Flag' else '' for col in s.index]

        styled_df = summary_df_reset.style.apply(hide_column, axis=0)
        html_table = styled_df.render()

        # Display the table with the hidden 'Priority_Flag' column
        # st.markdown(html_table, unsafe_allow_html=True)

        # Render in Streamlit
        st.markdown("**Audio Files Summary: Predicted Depression Severity Risk**")
        st.markdown(html_with_style, unsafe_allow_html=True)

        # Display the table with the new Priority column
     #   st.write("**Audio Files Summary: Predicted Depression Severity Risk**")
       # st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Compute the MFCC mean value (from your features)
        mfcc_mean_value = np.mean(st.session_state["audio_results"]["summary_df"]['MFCC_Mean'])
        #mfcc_mean_value = np.mean(mfcc_features)  # or specific coefficient

        # Map to severity and priority
        severity, priority_flag = map_severity_with_priority(mfcc_mean_value)

        drisk = severity  # or any other logic you prefer

        try:
            drisk = severity
        except Exception as e:
            drisk = f"Error: {str(e)}"

        if isinstance(drisk, str) and drisk.startswith("Error"):
            st.error(drisk)
        else:
            st.success(f"**Prediction:** {drisk} risk of depression")

       # st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Display severity prediction for the audio
       # st.info(f"Predicted Severity from MFCC Mean: {severity}")

        # Reset index for styling
        summary_df_reset = summary_df.reset_index(drop=True)
        
      #  st.dataframe(styled_df, use_container_width=True, hide_index=True)

        # Display summary table with predicted severity
       # st.write("**Audio Files Summary: Predicted Depression Severity Risk:**")
       # st.dataframe(styled_df, use_container_width=True, hide_index=True)

       
    else:
        st.info("Upload audio files to predict depressive risk")

    ##

    st.subheader("📋 Physiological Data")
    #st.info("Click **Regenerate Physiological Data** to load the pretrained model and generate predictions")
    ####
    # 
    
    if "arts" not in st.session_state:
        st.info("Click **Run Inference** to load the pretrained model and generate predictions")
    else:
        arts = st.session_state["arts"]
        num_patients = arts["TOTAL_N"]  # Get the number of patients
####

####        
        expected_samples = len(arts["pred_sample"])

        if "physio_data" not in st.session_state:
            st.session_state["physio_data"] = simulate_physiological_markers(
                #st.session_state.get("n_samples_ui", expected_samples),
                st.session_state.get("n_samples_ui", num_patients),
                breathing_range=(breathing_min, breathing_max),
                tapping_range=(tapping_min, tapping_max),
                heart_rate_range=(heart_rate_min, heart_rate_max)
            )

        #num_physio_samples = len(st.session_state["physio_data"])
        physio_data = st.session_state["physio_data"]

       
        #if not isinstance(st.session_state["physio_data"], list):
        #    st.session_state["physio_data"] = st.session_state["physio_data"].tolist()

        #num_physio_samples = len(st.session_state["physio_data"])
    
        if isinstance(physio_data, list):
            physio_data = np.array(physio_data)
    
        num_physio_samples = physio_data.shape[0] # if isinstance(physio_data, np.ndarray) else len(physio_data)
       # expected_samples = len(arts["pred_sample])
                               
        if num_physio_samples < expected_samples:
            padding = np.full((expected_samples - num_physio_samples, physio_data.shape[1]), None)
            physio_data = np.vstack([physio_data, padding])


        # Convert to DataFrame
        physio_df = pd.DataFrame(
            physio_data,
            columns=["Breathing Rate", "Tapping Rate", "Heart Rate"],
            index=range(1, expected_samples + 1)
        )

        formatted_df = physio_df.applymap(lambda x: f"{float(x):.2f}" if isinstance(x, (int, float)) else x)

        #formatted_df = physio_df.applymap(lambda x: f"{x:.2f}" if x is not None else "N/A")

        #formatted_df = physio_df.applymap(lambda x: f"{x:.2f}")
    
        #st.dataframe(formatted_df, use_container_width=True)

        html_physio_table = formatted_df.to_html(index=True, header=True, classes="custom-table", escape=False)

        # Custom CSS to align Patient ID to the left
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
    
        ####
        # Create a DataFrame
        #physio_df = pd.DataFrame(physio_data)
     
        # Extracting the average values for prediction
        breathing_rate = physio_df["Breathing Rate"].mean()
        tapping_rate = physio_df["Tapping Rate"].mean()
        heart_rate = physio_df["Heart Rate"].mean()

        # Classify dementia risk (using the example classify_dementia_risk function)
        risk = classify_dementia_risk(breathing_rate, tapping_rate, heart_rate)

                
        # Feature Distributions Subheader
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
            st.error(risk)  # Display error message in red
        else:
            st.success(f"**Prediction:** {risk}")  # Display success message in green

        # Physiological Data Subheader
      #  st.subheader("📋 Physiological Data")

        # Format the dataframe values to two decimal places
       # formatted_df = physio_df.applymap(lambda x: f"{x:.2f}")

        # Display the formatted dataframe in Streamlit
       # st.dataframe(formatted_df, use_container_width=True)

        
        ####
        #st.dataframe(st.session_state["physio_data"], use_container_width=True)  # Display as a dataframe
        # st.dataframe(physio_df, use_container_width=True)
        # formatted_df = physio_df.applymap(lambda x: f"{x:.2f}")
        # st.dataframe(formatted_df, use_container_width=True)

        st.subheader("Depression Severity: Inference Results")
    
    
       ###
    
        if "arts" not in st.session_state:
            st.info("Click **Run Inference** to load the pretrained model and generate predictions")
        else:
            arts = st.session_state["arts"]

            # arts = st.session_state["arts"]
    
            # Dataset Summary
            st.write(f"**Samples**: {arts['TOTAL_N']}  |  **Features**: {len(arts['feat_names'])}")

            # Misinformation Simulation for TOTAL_N
            S_list_, I_list_, R_list_, G_net_ = simulate_misinformation(
                num_nodes=arts["TOTAL_N"], trans_prob=trans_prob, rec_prob=rec_prob, steps=steps
            )
            misinfo_risk_ = I_list_[-1] / arts["TOTAL_N"]

            st.info(f"Misinformation risk applied: **{misinfo_risk_:.2f}**")
    
            # Adjusted severities + allocation
            # adjusted_all_ = arts["pred_sample"] * (1 - misinfo_risk_)
            # treated, untreated = allocate_resources(adjusted_all_, capacity=capacity)

            # Prepare adjusted severity
            expected_samples = len(arts["pred_sample"])
    
            #  Ensure physiological data matches expected size
            physio_data = st.session_state.get("physio_data")

            # Convert list to numpy array if necessary
            if isinstance(physio_data, list):
                physio_data = np.array(physio_data)

            if physio_data is not None and physio_data.shape[0] == arts["TOTAL_N"]:
                # Normalise physiological data (min-max)
                physio_min = physio_data.min(axis=0)
                physio_max = physio_data.max(axis=0)
                physio_norm = (physio_data - physio_min) / (physio_max - physio_min + 1e-6)

                # Weighted sum to create a physiological risk score
                weights = np.array([0.3, 0.3, 0.4])  # Adjust weights as desired
                physio_risk_score = np.clip(physio_norm @ weights, 0, 1)

                # Adjust severity based on physiological risk
                alpha = 0.1  # Control influence of physio impact on severity
                adjusted_all_ = arts["pred_sample"] * (1 - misinfo_risk_) * (1 + alpha * physio_risk_score)
                st.caption("Adjusted severity = Raw severity × (1 - misinformation risk) × (1 + α × physiological risk score)")
            else:
                # Fallback if physio data missing or mismatch e.g., 10 samples to inference 100
                adjusted_all_ = arts["pred_sample"] * (1 - misinfo_risk_)
                st.warning("⚠️ Physiological data mismatch. Falling back to severity adjusted by misinformation only. Increase the number of physiological samples to 100 include the data in severity results")

            treated, untreated = allocate_resources(adjusted_all_, capacity=capacity)


    
    # Get physiological data2
   # physio_data = st.session_state.get("physio_data")
   # if physio_data is not None and physio_data.shape[0] == arts["TOTAL_N"]:
        # Normalise physiological data (min-max)
   #     physio_min = physio_data.min(axis=0)
   #     physio_max = physio_data.max(axis=0)
   #     physio_norm = (physio_data - physio_min) / (physio_max - physio_min + 1e-6)

        # Weighted sum to create a physiological risk score
  #      weights = np.array([0.3, 0.3, 0.4])  # Adjust weights as desired
  #      physio_risk_score = np.clip(physio_norm @ weights, 0, 1)

  #      alpha = 0.1  # Control influence of physio data on severity
  #      adjusted_all_ = arts["pred_sample"] * (1 - misinfo_risk_) * (1 + alpha * physio_risk_score)
  #      st.caption("Adjusted severity = Raw severity × (1 - misinformation risk) × (1 + α × physiological risk score)")
  #  else:
        # Fallback if physio data missing or mismatch e.g., 10 samples to inference 100
  #      adjusted_all_ = arts["pred_sample"] * (1 - misinfo_risk_)
  #      st.warning("⚠️ Physiological data mismatch. Falling back to severity adjusted by misinformation only.")
  #  treated, untreated = allocate_resources(adjusted_all_, capacity=capacity)
    ####


    
####
    
            # Patient table
            treated_1_based = [i + 1 for i in treated]

            df_all = pd.DataFrame({
                "Patient ID": list(range(1, len(adjusted_all_)+1)),
                #"Raw Severity": np.round(arts["pred_sample"], 2),
                "Raw Severity": [f"{x:.2f}" for x in arts["pred_sample"]],
                #"Adjusted Severity": np.round(adjusted_all_, 2),
                "Adjusted Severity": [f"{x:.2f}" for x in adjusted_all_],
        
            })

            df_all["Priority"] = df_all["Patient ID"].apply(
                lambda pid: "✅ Yes" if pid in treated_1_based else "❌ No"
            )

            #df_all = df_all.reset_index(drop=True)
                                    
            html_table = df_all.head(100).to_html(index=False, classes="custom-table")
                                        

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
            html_table = df_all.head(100).to_html(index=False, classes="custom-table")

            st.markdown(custom_css, unsafe_allow_html=True)
            st.markdown(html_table, unsafe_allow_html=True)
    
            #st.markdown(custom_css + html_table, unsafe_allow_html=True)
    
            #st.write(styled_df.to_html(), unsafe_allow_html=True)
    
            #st.dataframe(df_all.drop(columns=["Patient ID"]).head(100), use_container_width=True, hide_index=True)
    
            #st.dataframe(df_all.head(100), use_container_width=True, hide_index=true)

   

            # Render the styled table
            #st.markdown(custom_css + html, unsafe_allow_html=True)

    
            # Patient Details & Explanations
            st.subheader("📊 Patient Details and Explanations")
            patient_idx = st.selectbox("Select Patient ID:", options=list(range(1,len(adjusted_all_)+1)), index=0)
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

            # Explanation block (LIME default like your sketch; SHAP optional)
            if method == "LIME":
                st.subheader("LIME Explanation")
                lime_exp = arts["explainer_lime"].explain_instance(
                    arts["X_sample_s"][patient_idx],
                    arts["model"].predict,
                    num_features=min(10, len(arts["feat_names"]))
                )
        
                fig = lime_exp.as_pyplot_figure()
                ax = fig.gca()
                feature_weights = lime_exp.as_list()
        
                # Apply custom color scheme to LIME chart
                # custom_colors = ['#003A6B', '#1B5886', '#3776A1', '#5293BB', '#6EB1D6', '#89CFF1']

                color_increase = '#3776A1'
                color_decrease = '#6EB1D6'

                # Get feature weights directly from the LIME explanation
       
                bars = ax.patches

                for bar, (feature, weight) in zip(bars, feature_weights):
                    bar.set_color(color_increase if weight >= 0 else color_decrease)
                    bar.set_alpha(0.8)
 #                if weight >= 0:
 #                   bar.set_color(color_increase)
 #               else:
 #                   bar.set_color(color_decrease)
 #               bar.set_alpha(0.8)


            #    bar.set_color(custom_colors[color_idx])
            #    bar.set_alpha(0.8)  # Add some transparency for better aesthetics

                increase_patch = mpatches.Patch(color=color_increase, label='↑ Increases PHQ-8 Score')
                decrease_patch = mpatches.Patch(color=color_decrease, label='↓ Decreases PHQ-8 Score')
                ax.legend(handles=[increase_patch, decrease_patch], loc='lower left', bbox_to_anchor=(0, 0), title="Feature Effect")
        
            # Update the figure style
                fig.patch.set_facecolor('white')
                ax.set_facecolor('#f8f9fa')  # Light background
        
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            elif method == "SHAP":
                st.subheader("SHAP Explanation")
                shap_values = arts["explainer_shap"].shap_values(arts["X_sample_s"][patient_idx:patient_idx+1])
                if isinstance(shap_values, list):
            # Assuming binary classification, use the SHAP values for the positive class (index 1)
                    shap_values_local = shap_values[1]
                else:
                    shap_values_local = shap_values

        # Round SHAP values and feature values to 2 decimals
                shap_values_rounded = np.round(shap_values_local, 2)
                features_rounded = np.round(arts["X_sample_s"][patient_idx:patient_idx+1], 2)

                shap_value_display = {
                    f"Feature {i}": f"{shap_values_rounded[0][i]:.2f}"  # Accessing the individual value within the inner array
                    for i in range(len(shap_values_rounded[0]))
                }

                shap.force_plot(
                    arts["explainer_shap"].expected_value,  # Expected value
                    shap_values_rounded[0],  # Rounded SHAP values for the selected instance (access the first instance)
                    features=features_rounded[0],  # Feature values for the selected instance
                    matplotlib=True,  # Using Matplotlib for plotting
                    show=False  # Don't show the plot immediately, we'll customize it
                )
        
                fig_local = plt.gcf()
                ax = plt.gca()

                ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{x:.2f}"))
                ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, pos: f"{y:.2f}"))

                for tick in ax.get_xticklabels():
                    tick.set_rotation(0)
                    tick.set_fontsize(10)
                for tick in ax.get_yticklabels():
                    tick.set_rotation(0)
                    tick.set_fontsize(10)
        
                st.pyplot(fig_local, use_container_width=True)
                plt.close(fig_local)

        # Misinformation Spread Over Time
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

        # Network Snapshot
        st.subheader("🌐 Social Network Visualisation: Final Network State")
        fig_net, ax_net = plt.subplots(figsize=(7, 5))
        pos = nx.spring_layout(G_net_, seed=42)
        c_map = {'S': '#003A6B', 'I': '#3776A1', 'R': '#89CFF1'}
        label_map = {'S': 'Susceptible', 'I': 'Infected', 'R': 'Recovered'}
    
        node_colors = [c_map[G_net_.nodes[n]['state']] for n in G_net_.nodes()]
        nx.draw(G_net_, pos, node_color=node_colors, node_size=20, with_labels=False, ax=ax_net, edge_color='#414141')
    
        # Create legend patches
        legend_patches = [mpatches.Patch(color=color, label=label_map[state]) for state, color in c_map.items()]
        ax_net.legend(handles=legend_patches, title="Node State", loc='best')
    
        st.pyplot(fig_net, use_container_width=True)
        plt.close(fig_net)
#####
def main():
    if "streamlit" in sys.argv[0].lower() or os.environ.get("STREAMLIT_SERVER_PORT"):
        run_app()
    else:
        main_cli()  # Ensure main_cli() exists

if __name__ == "__main__":
    main()

