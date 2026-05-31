import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import loadmat
import tensorflow as tf
import joblib
import tempfile
import os


st.set_page_config(
    page_title="EEG ADHD Detection Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 1.5rem 1.8rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0f766e 100%);
            color: white;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.22);
            margin-bottom: 1.25rem;
        }
        .hero h1 {
            font-size: 2.2rem;
            margin-bottom: 0.3rem;
        }
        .hero p {
            margin-bottom: 0;
            opacity: 0.92;
            font-size: 1rem;
        }
        .info-card {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
            height: 100%;
        }
        .step-card {
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 16px;
            padding: 1rem;
            height: 100%;
        }
        .result-card {
            border-radius: 20px;
            padding: 1.2rem 1.3rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            background: #ffffff;
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.07);
        }
        .result-positive {
            background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%);
            border-left: 6px solid #dc2626;
        }
        .result-negative {
            background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
            border-left: 6px solid #16a34a;
        }
        .result-meter {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1.1;
            margin: 0.15rem 0 0.35rem 0;
        }
        .muted {
            color: #475569;
        }
        .small-note {
            font-size: 0.9rem;
            color: #64748b;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def load_assets():
    model = tf.keras.models.load_model("model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_assets()

# =============================
# UI
# =============================
st.markdown(
    """
    <div class="hero">
        <h1>EEG-based ADHD Detection Dashboard</h1>
        <p>Upload EEG data, follow the analysis flow, and review a step-by-step explanation of the prediction result.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================
# Required EEG channels (19)
# =============================
REQUIRED_CHANNELS = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz'
]

CHANNEL_INDEX = {name: idx for idx, name in enumerate(REQUIRED_CHANNELS)}
LEFT_HEMISPHERE_CHANNELS = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T7', 'P7']
RIGHT_HEMISPHERE_CHANNELS = ['Fp2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T8', 'P8']
LEFT_HEMISPHERE_IDX = [CHANNEL_INDEX[name] for name in LEFT_HEMISPHERE_CHANNELS]
RIGHT_HEMISPHERE_IDX = [CHANNEL_INDEX[name] for name in RIGHT_HEMISPHERE_CHANNELS]

# =============================
# Windowing parameters
# =============================
WINDOW_SIZE = 256
STEP_SIZE = 128

# =============================
# Helper functions
# =============================
def create_windows(eeg):
    """eeg shape: (channels, time)"""
    windows = []
    for i in range(0, eeg.shape[1] - WINDOW_SIZE + 1, STEP_SIZE):
        windows.append(eeg[:, i:i + WINDOW_SIZE])
    return np.array(windows)

def load_mat_file(file):
    mat = loadmat(file)
    key = [k for k in mat.keys() if not k.startswith("__")][0]
    eeg = mat[key].T
    return eeg

def load_csv_or_txt(file):
    try:
        df = pd.read_csv(file)
        df = df.select_dtypes(include=[np.number])
        eeg = df.values.T
        return eeg
    except Exception:
        file.seek(0)
        data = np.loadtxt(file)
        eeg = data.T
        return eeg


def score_to_percentages(score):
    adhd_percent = max(0.0, min(100.0, (1.0 - score) * 100.0))
    control_percent = 100.0 - adhd_percent
    return adhd_percent, control_percent


def predict_eeg_score(eeg):
    windows = create_windows(eeg)

    if len(windows) == 0:
        raise ValueError("Not enough data to create a prediction window.")

    reshaped = windows.reshape(-1, windows.shape[-1])
    scaled = scaler.transform(reshaped)
    scaled = scaled.reshape(windows.shape)

    X = scaled[..., np.newaxis]
    preds = model.predict(X, verbose=0)

    return {
        "score": float(np.mean(preds)),
        "confidence": float(abs(np.mean(preds) - 0.5) * 2 * 100),
        "windows": int(windows.shape[0]),
    }


def mask_hemisphere(eeg, active_indices):
    masked = np.zeros_like(eeg)
    masked[active_indices, :] = eeg[active_indices, :]
    return masked


def render_step(title, description, detail):
    st.markdown(
        f"""
        <div class="step-card">
            <h4 style="margin-bottom:0.35rem;">{title}</h4>
            <div class="muted" style="margin-bottom:0.45rem;">{description}</div>
            <div class="small-note">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(is_adhd, mean_pred, confidence):
    adhd_percent, control_percent = score_to_percentages(mean_pred)
    if is_adhd:
        status_text = "ADHD Detected"
        card_class = "result-card result-positive"
        explanation = (
            "The model's score is below the decision threshold of 0.50, so the EEG pattern is closer to the ADHD side of the trained classifier. "
            "This means the sample shows characteristics the model learned to associate with ADHD in the training data."
        )
        clinical_note = (
            "This is a screening result, not a medical diagnosis. A low score should be reviewed together with clinical history, behavioral assessment, and a qualified specialist's opinion."
        )
    else:
        status_text = "No ADHD Detected"
        card_class = "result-card result-negative"
        explanation = (
            "The model's score is at or above the decision threshold of 0.50, so the EEG pattern is closer to the non-ADHD side of the trained classifier. "
            "This suggests the sample does not strongly resemble the ADHD patterns seen during training."
        )
        clinical_note = (
            "This does not rule out ADHD in a clinical sense. EEG alone cannot replace a full diagnostic evaluation, especially when symptoms, age, or recording conditions vary."
        )

    st.markdown(
        f"""
        <div class="{card_class}">
            <div class="small-note">Model result</div>
            <div class="result-meter">{status_text}</div>
            <div class="muted">Mean prediction score: <strong>{mean_pred:.4f}</strong></div>
            <div class="muted">ADHD likelihood: <strong>{adhd_percent:.2f}%</strong></div>
            <div class="muted">Non-ADHD likelihood: <strong>{control_percent:.2f}%</strong></div>
            <div class="muted">Confidence: <strong>{confidence:.2f}%</strong></div>
            <p style="margin-top:0.85rem; margin-bottom:0.5rem;">{explanation}</p>
            <p style="margin-bottom:0;">{clinical_note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_summary(label, score, confidence, windows, positive_label, negative_label):
    adhd_percent, control_percent = score_to_percentages(score)
    is_positive = score < 0.5
    state_text = positive_label if is_positive else negative_label
    state_class = "result-positive" if is_positive else "result-negative"
    st.markdown(
        f"""
        <div class="result-card {state_class}">
            <div class="small-note">{label}</div>
            <div class="result-meter">{state_text}</div>
            <div class="muted">Mean score: <strong>{score:.4f}</strong></div>
            <div class="muted">ADHD likelihood: <strong>{adhd_percent:.2f}%</strong></div>
            <div class="muted">Non-ADHD likelihood: <strong>{control_percent:.2f}%</strong></div>
            <div class="muted">Confidence: <strong>{confidence:.2f}%</strong></div>
            <div class="muted">Windows analyzed: <strong>{windows}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_post_result_explanation(score, mode_label):
    adhd_percent, control_percent = score_to_percentages(score)
    if score < 0.5:
        st.markdown(
            f"""
            <div class="info-card">
                <p><strong>ADHD Detected</strong> means the model score fell below 0.50 in {mode_label} mode.</p>
                <p><strong>Percent view:</strong> ADHD likelihood is {adhd_percent:.2f}% and non-ADHD likelihood is {control_percent:.2f}%.</p>
                <p><strong>How to read it:</strong> this is a screening output. It should be interpreted with symptoms, history, and a clinician's assessment.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="info-card">
                <p><strong>No ADHD Detected</strong> means the model score was 0.50 or higher in {mode_label} mode.</p>
                <p><strong>Percent view:</strong> ADHD likelihood is {adhd_percent:.2f}% and non-ADHD likelihood is {control_percent:.2f}%.</p>
                <p><strong>How to read it:</strong> this does not fully rule out ADHD. It only means this sample was not strongly classified as ADHD by the model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.sidebar.header("Dashboard Workflow")
st.sidebar.write("Use the steps below to understand how the analysis moves from raw EEG to the final result.")
st.sidebar.markdown(
    """
    1. Upload a MAT, EDF, CSV, or TXT file.
    2. The app checks channel count and recording length.
    3. The EEG is split into overlapping windows.
    4. Each window is scaled using the saved scaler.
    5. The trained model predicts a score for every window.
    6. The final result is the average score across all windows.
    7. The dashboard explains what the result means.
    """
)

st.sidebar.divider()
st.sidebar.subheader("Model Rules")
st.sidebar.write(f"Required channels: {len(REQUIRED_CHANNELS)}")
st.sidebar.write(f"Window size: {WINDOW_SIZE} samples")
st.sidebar.write(f"Step size: {STEP_SIZE} samples")

st.header("Upload EEG Data")
control_cols = st.columns([0.7, 0.3])
with control_cols[0]:
    file_type = st.selectbox(
        "Select EEG file type",
        ["MAT", "EDF", "CSV / TXT"],
        help="Choose the format that matches the file you are uploading.",
    )
    detection_mode = st.radio(
        "Select detection mode",
        ["Total EEG", "Hemisphere-wise"],
        horizontal=True,
        help="Total EEG uses all 19 channels. Hemisphere-wise compares left and right hemisphere channel groups.",
    )
with control_cols[1]:
    st.markdown(
        """
        <div class="info-card">
            <p style="margin-bottom:0.35rem;"><strong>Accepted formats</strong></p>
            <p class="small-note" style="margin-bottom:0;">MAT, EDF, CSV, TXT</p>
            <p class="small-note" style="margin-bottom:0;">The file must contain 19 EEG channels.</p>
            <p class="small-note" style="margin-bottom:0;">Hemisphere-wise mode shows left and right side scores separately.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

uploaded_file = st.file_uploader(
    "Upload EEG file",
    type=["mat", "edf", "csv", "txt"],
    key="eeg_file_uploader",
)

# =============================
# Main logic
# =============================
if uploaded_file is not None:
    try:
        if file_type == "MAT":
            eeg = load_mat_file(uploaded_file)

        elif file_type == "EDF":
            # Save EDF temporarily (Streamlit fix)
            import mne

            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
            os.remove(temp_path)

            missing = set(REQUIRED_CHANNELS) - set(raw.ch_names)
            if missing:
                st.error(f"Missing required channels: {missing}")
                st.stop()

            raw.pick_channels(REQUIRED_CHANNELS)
            raw.resample(128)
            eeg = raw.get_data()

        else:  # CSV / TXT
            eeg = load_csv_or_txt(uploaded_file)

        if eeg.shape[0] != 19:
            st.error("EEG must have exactly 19 channels")
            st.stop()

        if eeg.shape[1] < WINDOW_SIZE:
            st.error("EEG recording too short for prediction")
            st.stop()

        if st.button("Predict ADHD"):
            if detection_mode == "Total EEG":
                total_result = predict_eeg_score(eeg)
                total_score = total_result["score"]
                total_confidence = total_result["confidence"]
                total_adhd_percent, total_control_percent = score_to_percentages(total_score)

                result_cols = st.columns([1.05, 0.95])
                with result_cols[0]:
                    render_result_card(total_score < 0.5, total_score, total_confidence)
                with result_cols[1]:
                    st.markdown(
                        """
                        <div class="info-card">
                            <p><strong>Total EEG mode</strong></p>
                            <p class="small-note">Uses all 19 channels together for one final screening score.</p>
                            <p class="small-note">This is the fastest path when you want a single overall result.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.metric("Average model score", f"{total_score:.4f}")
                    st.metric("ADHD likelihood", f"{total_adhd_percent:.2f}%")
                    st.metric("Non-ADHD likelihood", f"{total_control_percent:.2f}%")
                    st.metric("Confidence", f"{total_confidence:.2f}%")
                    st.metric("Windows analyzed", total_result["windows"])
                    st.progress(min(max(total_adhd_percent / 100, 0.0), 1.0))

                st.markdown("### Result explanation")
                render_post_result_explanation(total_score, "total EEG")

            else:
                left_eeg = mask_hemisphere(eeg, LEFT_HEMISPHERE_IDX)
                right_eeg = mask_hemisphere(eeg, RIGHT_HEMISPHERE_IDX)

                left_result = predict_eeg_score(left_eeg)
                right_result = predict_eeg_score(right_eeg)
                combined_score = float((left_result["score"] + right_result["score"]) / 2)
                combined_confidence = abs(combined_score - 0.5) * 2 * 100
                combined_adhd_percent, combined_control_percent = score_to_percentages(combined_score)
                left_adhd_percent, left_control_percent = score_to_percentages(left_result["score"])
                right_adhd_percent, right_control_percent = score_to_percentages(right_result["score"])

                result_cols = st.columns([1.05, 0.95])
                with result_cols[0]:
                    render_result_card(combined_score < 0.5, combined_score, combined_confidence)
                with result_cols[1]:
                    st.markdown(
                        """
                        <div class="info-card">
                            <p><strong>Hemisphere-wise mode</strong></p>
                            <p class="small-note">Evaluates the left and right hemisphere channel groups separately, then averages the two scores.</p>
                            <p class="small-note">This shows whether one side contributes more strongly than the other.</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.metric("Combined score", f"{combined_score:.4f}")
                    st.metric("ADHD likelihood", f"{combined_adhd_percent:.2f}%")
                    st.metric("Non-ADHD likelihood", f"{combined_control_percent:.2f}%")
                    st.metric("Confidence", f"{combined_confidence:.2f}%")
                    st.metric("Left windows", left_result["windows"])
                    st.metric("Right windows", right_result["windows"])

                sub_cols = st.columns(2)
                with sub_cols[0]:
                    render_result_summary(
                        "Left hemisphere",
                        left_result["score"],
                        abs(left_result["score"] - 0.5) * 2 * 100,
                        left_result["windows"],
                        f"Left side looks ADHD-like ({left_adhd_percent:.2f}%)",
                        f"Left side looks non-ADHD-like ({left_control_percent:.2f}%)",
                    )
                with sub_cols[1]:
                    render_result_summary(
                        "Right hemisphere",
                        right_result["score"],
                        abs(right_result["score"] - 0.5) * 2 * 100,
                        right_result["windows"],
                        f"Right side looks ADHD-like ({right_adhd_percent:.2f}%)",
                        f"Right side looks non-ADHD-like ({right_control_percent:.2f}%)",
                    )

                st.markdown("### Result explanation")
                render_post_result_explanation(combined_score, "hemisphere-wise")

    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Upload an EEG file to start the dashboard workflow.")
