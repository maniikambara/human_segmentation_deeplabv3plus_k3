import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from inference import segment_image

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "files", "model.h5")

BG = "#F6F3EC"
CARD = "#FFFFFF"
INK = "#211D1A"
MUTED = "#7A736A"
ACCENT = "#C4562E"
LINE = "#E7E1D5"

st.set_page_config(page_title="Segmentasi Manusia", page_icon=None, layout="centered")

st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        #MainMenu, footer, header {{ visibility: hidden; }}
        .block-container {{ padding-top: 2.6rem; padding-bottom: 3rem; max-width: 760px; }}
        html, body, [class*="st-"], .stMarkdown, p, span, div {{
            font-family: 'Inter', sans-serif;
        }}
        .stApp {{ background: {BG}; color: {INK}; }}

        .brand {{
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 1.55rem;
            letter-spacing: -0.02em;
            color: {INK};
            margin-bottom: 0.1rem;
        }}
        .brand span {{ color: {ACCENT}; }}
        .tagline {{
            color: {MUTED};
            font-size: 0.98rem;
            margin-bottom: 2.1rem;
        }}

        [data-testid="stFileUploadDropzone"] {{
            background: {CARD};
            border: 1.5px dashed #D8CFBE;
            border-radius: 14px;
            color: {INK};
            transition: border-color 0.15s ease;
        }}
        [data-testid="stFileUploadDropzone"]:hover {{
            border-color: {ACCENT};
        }}
        [data-testid="stFileUploadDropzone"] small {{
            color: {MUTED};
        }}
        [data-testid="stFileUploadDropzone"] button {{
            background: {INK};
            color: {BG};
            border: none;
            border-radius: 8px;
            font-weight: 500;
        }}
        [data-testid="stFileUploadDropzone"] button:hover {{
            background: {ACCENT};
            color: white;
        }}

        div[role="radiogroup"] {{
            gap: 0.4rem;
        }}
        div[role="radiogroup"] label {{
            background: {CARD};
            border: 1px solid {LINE};
            padding: 0.32rem 0.95rem;
            border-radius: 999px;
            margin-right: 0;
            transition: all 0.15s ease;
        }}
        div[role="radiogroup"] label:hover {{
            border-color: {ACCENT};
        }}
        div[role="radiogroup"] input:checked + div {{
            color: {ACCENT} !important;
            font-weight: 600;
        }}

        [data-testid="column"] [data-testid="stVerticalBlock"] {{
            background: {CARD};
            border: 1px solid {LINE};
            border-radius: 16px;
            padding: 1rem 0 0 0;
            margin-top: 1rem;
            overflow: hidden;
        }}
        .result-label {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {MUTED};
            font-weight: 600;
            padding: 0 1.1rem 0.8rem 1.1rem;
        }}
        .stat-row {{
            display: flex;
            gap: 1.6rem;
            margin-top: 0.9rem;
            padding-top: 0.9rem;
            border-top: 1px solid {LINE};
        }}
        .stat {{ font-size: 0.86rem; color: {MUTED}; }}
        .stat b {{ color: {INK}; font-weight: 600; }}

        [data-testid="stImage"] img {{
            display: block;
        }}

        .empty-note {{
            border: 1px solid {LINE};
            background: {CARD};
            border-radius: 14px;
            padding: 2.4rem 1.5rem;
            text-align: center;
            color: {MUTED};
            font-size: 0.92rem;
            margin-top: 0.6rem;
        }}

        .missing-model {{
            background: {CARD};
            border: 1px solid {LINE};
            border-left: 3px solid {ACCENT};
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            font-size: 0.92rem;
            color: {INK};
        }}
        .missing-model code {{
            background: {BG};
            padding: 0.1rem 0.4rem;
            border-radius: 5px;
            font-size: 0.85rem;
        }}

        div.stSlider {{ padding-top: 0.2rem; }}
        [data-testid="stDownloadButton"] button {{
            background: transparent;
            border: 1px solid {LINE};
            color: {INK};
            border-radius: 8px;
            font-weight: 500;
        }}
        [data-testid="stDownloadButton"] button:hover {{
            border-color: {ACCENT};
            color: {ACCENT};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    import tensorflow as tf
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def to_bgr(uploaded_file):
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def render_overlay(image_bgr, mask, strength):
    accent = np.array([46, 86, 196], dtype=np.float32)  # BGR of ACCENT (#C4562E)
    out = image_bgr.astype(np.float32)
    m = mask.astype(bool)
    out[m] = out[m] * (1 - strength) + accent * strength
    return out.astype(np.uint8)


def render_cutout(image_bgr, mask, backdrop_bgr):
    out = np.full_like(image_bgr, backdrop_bgr, dtype=np.uint8)
    m = mask.astype(bool)
    out[m] = image_bgr[m]
    return out


def render_mask(mask):
    return (mask * 255).astype(np.uint8)


def bgr_to_png_bytes(image_bgr_or_gray):
    ok, buf = cv2.imencode(".png", image_bgr_or_gray)
    return buf.tobytes() if ok else None


st.markdown('<div class="brand">Segmentasi<span>.</span></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tagline">Unggah satu foto berisi orang &mdash; garis besarnya langsung terlihat.</div>',
    unsafe_allow_html=True,
)

model = load_model()

if model is None:
    st.markdown(
        f"""
        <div class="missing-model">
        Checkpoint model belum ditemukan di <code>files/model.h5</code>.
        Jalankan <code>train.ipynb</code> sampai selesai terlebih dahulu, lalu muat ulang halaman ini.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

uploaded = st.file_uploader(
    "Unggah gambar", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

if uploaded is None:
    st.markdown(
        '<div class="empty-note">Belum ada gambar. Seret foto ke sini atau klik untuk memilih file.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

image_bgr = to_bgr(uploaded)

with st.spinner("Memproses…"):
    mask = segment_image(model, image_bgr)

fg_ratio = float(mask.mean())

view = st.radio(
    "Tampilan",
    ["Overlay", "Potongan", "Mask"],
    horizontal=True,
    label_visibility="collapsed",
)

if view == "Overlay":
    strength = st.slider("Kekuatan warna", 0.0, 1.0, 0.55, 0.05, label_visibility="collapsed")
    result_bgr = render_overlay(image_bgr, mask, strength)
    caption = "Overlay"
elif view == "Potongan":
    result_bgr = render_cutout(image_bgr, mask, (238, 233, 221))
    caption = "Potongan (latar dihapus)"
else:
    result_bgr = render_mask(mask)
    caption = "Mask biner"

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="result-label">Asli</div>', unsafe_allow_html=True)
    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

with col2:
    st.markdown(f'<div class="result-label">{caption}</div>', unsafe_allow_html=True)
    if view == "Mask":
        st.image(result_bgr, use_column_width=True, clamp=True)
    else:
        st.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

st.markdown(
    f"""
    <div class="stat-row">
        <div class="stat">Resolusi &nbsp;<b>{image_bgr.shape[1]}&times;{image_bgr.shape[0]}</b></div>
        <div class="stat">Piksel foreground &nbsp;<b>{fg_ratio * 100:.1f}%</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

png_bytes = bgr_to_png_bytes(result_bgr)
if png_bytes:
    st.download_button(
        "Unduh hasil",
        data=png_bytes,
        file_name=f"{os.path.splitext(uploaded.name)[0]}_{view.lower()}.png",
        mime="image/png",
    )
