import os
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

from inference import segment_image

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "files", "model.h5")
SAMPLES_DIR = os.path.join(APP_DIR, "test_images", "samples")
SAMPLE_FILES = ["sample1.jpg", "sample2.jpg", "sample3.jpg"]

BG = "#FFFFFF"
SURFACE = "#F8F9FB"
CARD = "#FFFFFF"
INK = "#111827"
MUTED = "#6B7280"
ACCENT = "#2563EB"
LINE = "#E5E7EB"

st.set_page_config(page_title="Segmentasi Manusia", page_icon="🧍", layout="centered")


def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        #MainMenu, footer, header {{ visibility: hidden; }}
        .block-container {{ padding-top: 2.4rem; padding-bottom: 3rem; max-width: 780px; }}
        html, body, [class*="st-"], .stMarkdown, p, span, div {{
            font-family: 'Inter', sans-serif;
        }}
        .stApp {{ background: {BG}; color: {INK}; }}

        [data-testid="stIconMaterial"] {{
            font-family: 'Material Symbols Rounded', sans-serif !important;
        }}

        .topbar {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.6rem;
        }}
        .brand {{
            font-weight: 800;
            font-size: 1.25rem;
            letter-spacing: -0.01em;
            color: {INK};
        }}
        .brand span {{ color: {ACCENT}; }}

        .hero-title {{
            font-weight: 800;
            font-size: 2.5rem;
            line-height: 1.15;
            letter-spacing: -0.02em;
            color: {INK};
            text-align: center;
            margin-bottom: 0.9rem;
        }}
        .hero-sub {{
            color: {MUTED};
            font-size: 1.05rem;
            text-align: center;
            max-width: 520px;
            margin: 0 auto 2.2rem auto;
        }}
        .badge {{
            display: inline-block;
            background: {SURFACE};
            border: 1px solid {LINE};
            color: {MUTED};
            font-size: 0.78rem;
            font-weight: 600;
            padding: 0.3rem 0.8rem;
            border-radius: 999px;
            margin-bottom: 1.1rem;
        }}
        .badge b {{ color: {ACCENT}; }}
        .center {{ text-align: center; }}

        [data-testid="stFileUploaderDropzone"], [data-testid="stFileUploadDropzone"] {{
            background: {SURFACE};
            border: 1.5px dashed #C7CDD6;
            border-radius: 14px;
            color: {INK};
            transition: border-color 0.15s ease;
        }}
        [data-testid="stFileUploaderDropzone"]:hover, [data-testid="stFileUploadDropzone"]:hover {{
            border-color: {ACCENT};
        }}
        [data-testid="stFileUploadDropzone"] button, [data-testid="stFileUploaderDropzone"] button {{
            background: {ACCENT};
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 500;
        }}

        div[role="radiogroup"] {{
            gap: 0.4rem;
        }}
        div[role="radiogroup"] label {{
            background: {CARD};
            border: 1px solid {LINE};
            padding: 0.35rem 0.95rem;
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

        .samples-label {{
            text-align: center;
            font-size: 0.85rem;
            color: {MUTED};
            font-weight: 600;
            margin: 1.6rem 0 0.8rem 0;
        }}
        [data-testid="stImage"] {{
            border-radius: 10px;
            overflow: hidden;
        }}
        div[data-testid="column"] .stButton button {{
            width: 100%;
            background: {CARD};
            border: 1px solid {LINE};
            color: {MUTED};
            font-size: 0.8rem;
            border-radius: 7px;
            padding: 0.25rem 0;
        }}
        div[data-testid="column"] .stButton button:hover {{
            border-color: {ACCENT};
            color: {ACCENT};
        }}

        .panel-title {{
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: {MUTED};
            font-weight: 600;
            margin-bottom: 0.6rem;
        }}

        [data-testid="stMetric"] {{
            background: {SURFACE};
            border: 1px solid {LINE};
            border-radius: 10px;
            padding: 0.8rem 1rem;
        }}
        [data-testid="stMetricLabel"] {{ color: {MUTED}; }}

        .missing-model {{
            background: {SURFACE};
            border-left: 3px solid {ACCENT};
            border-radius: 8px;
            padding: 1rem 1.2rem;
            font-size: 0.92rem;
            color: {INK};
        }}
        .missing-model code {{
            background: {CARD};
            border: 1px solid {LINE};
            padding: 0.1rem 0.4rem;
            border-radius: 5px;
            font-size: 0.85rem;
        }}

        [data-testid="stDownloadButton"] button {{
            background: {ACCENT};
            border: none;
            color: white;
            border-radius: 6px;
            font-weight: 500;
        }}
        [data-testid="stDownloadButton"] button:hover {{ opacity: 0.9; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    import tensorflow as tf
    # Tanpa ini, TF mencoba mengalokasikan sebagian besar VRAM yang saat itu
    # bebas dalam satu alokasi besar saat sesi pertama kali menyentuh GPU.
    # Di GPU 6 GB yang juga dipakai kernel Jupyter lain (train/eval/predict.ipynb
    # yang masih berjalan), ini bisa gagal dengan "DNN library is not found" /
    # "Could not create cudnn handle" karena tidak ada cukup memori tersisa
    # untuk cuDNN. Sama seperti configure_gpu() di train.ipynb.
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass  # sudah diinisialisasi
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def to_bgr(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def render_overlay(image_bgr, mask, strength, color_bgr):
    color = np.array(color_bgr, dtype=np.float32)
    out = image_bgr.astype(np.float32)
    m = mask.astype(bool)
    out[m] = out[m] * (1 - strength) + color * strength
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


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


if "active_bytes" not in st.session_state:
    st.session_state.active_bytes = None
    st.session_state.active_name = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

model = load_model()

st.markdown(
    """
    <div class="topbar">
        <div class="brand">Segmentasi<span>.</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.markdown(
        """
        <div class="missing-model">
        Checkpoint model belum ditemukan di <code>files/model.h5</code>.
        Jalankan <code>train.ipynb</code> sampai selesai terlebih dahulu, lalu muat ulang halaman ini.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ---------- Landing page (no active image yet) ----------
if st.session_state.active_bytes is None:
    st.markdown('<div class="center"><span class="badge">Deep Learning &middot; <b>DeepLabV3+</b></span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Hapus &amp; Analisis Latar<br>Belakang Foto Manusia</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Model segmentasi semantik mengenali piksel manusia dalam foto secara otomatis &mdash; hasilnya bisa dijadikan overlay, potongan tanpa latar, atau mask biner.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Unggah gambar",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded is not None:
        st.session_state.active_bytes = uploaded.getvalue()
        st.session_state.active_name = uploaded.name
        rerun()

    def is_valid_image(path, min_size=100):
        try:
            with Image.open(path) as img:
                img.verify()
            with Image.open(path) as img:
                img.load()
                if img.width < min_size or img.height < min_size:
                    return False
            return True
        except Exception:
            return False

    available_samples = [
        f for f in SAMPLE_FILES
        if os.path.exists(os.path.join(SAMPLES_DIR, f)) and is_valid_image(os.path.join(SAMPLES_DIR, f))
    ]
    if available_samples:
        st.markdown('<div class="samples-label">atau coba salah satu contoh berikut</div>', unsafe_allow_html=True)
        cols = st.columns(len(available_samples))
        for col, fname in zip(cols, available_samples):
            path = os.path.join(SAMPLES_DIR, fname)
            with col:
                st.image(path, use_column_width=True)
                if st.button("Pakai foto ini", key=f"sample_{fname}"):
                    with open(path, "rb") as f:
                        st.session_state.active_bytes = f.read()
                    st.session_state.active_name = fname
                    rerun()
    else:
        st.markdown(
            f'<div class="samples-label">Taruh 3 foto di <code>test_images/samples/</code> '
            f'({", ".join(SAMPLE_FILES)}) agar muncul sebagai contoh di sini.</div>',
            unsafe_allow_html=True,
        )
    st.stop()

# ---------- Workspace (image active) ----------
if st.button("← Ganti gambar"):
    st.session_state.active_bytes = None
    st.session_state.active_name = None
    st.session_state.uploader_key += 1
    rerun()

image_bgr = to_bgr(st.session_state.active_bytes)

import tensorflow as tf  # sudah diimpor & di-cache oleh load_model(); murah di sini

try:
    with st.spinner("Memproses…"):
        mask = segment_image(model, image_bgr)
except tf.errors.OpError as e:
    st.markdown(
        f"""
        <div class="missing-model">
        Gagal menjalankan model di GPU (<code>{type(e).__name__}</code>).
        Ini biasanya terjadi kalau notebook Jupyter lain (train/eval/predict.ipynb)
        masih berjalan di background dan menahan VRAM &mdash; GPU 6&nbsp;GB tidak
        cukup untuk semuanya sekaligus. Matikan kernel notebook yang tidak
        dipakai (Kernel &rarr; Shut Down di Jupyter), lalu muat ulang halaman ini.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

fg_ratio = float(mask.mean())

view = st.radio(
    "Tampilan",
    ["Overlay", "Potongan", "Mask"],
    horizontal=True,
    label_visibility="collapsed",
)

strength, overlay_color, bg_color = 0.55, "#2563EB", "#EEE9DD"
if view == "Overlay":
    c1, c2 = st.columns([1, 2])
    with c1:
        overlay_color = st.color_picker("Warna overlay", overlay_color)
    with c2:
        strength = st.slider("Kekuatan warna", 0.0, 1.0, 0.55, 0.05)
elif view == "Potongan":
    bg_color = st.color_picker("Warna latar", bg_color)

if view == "Overlay":
    result_bgr = render_overlay(image_bgr, mask, strength, hex_to_bgr(overlay_color))
    caption = "Overlay"
elif view == "Potongan":
    result_bgr = render_cutout(image_bgr, mask, hex_to_bgr(bg_color))
    caption = "Potongan (latar dihapus)"
else:
    result_bgr = render_mask(mask)
    caption = "Mask biner"

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="panel-title">Asli</div>', unsafe_allow_html=True)
    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

with col2:
    st.markdown(f'<div class="panel-title">{caption}</div>', unsafe_allow_html=True)
    if view == "Mask":
        st.image(result_bgr, use_column_width=True, clamp=True)
    else:
        st.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

st.write("")
m1, m2, m3 = st.columns(3)
m1.metric("Resolusi", f"{image_bgr.shape[1]}×{image_bgr.shape[0]}")
m2.metric("Piksel foreground", f"{fg_ratio * 100:.1f}%")
m3.metric("Mode aktif", view)

png_bytes = bgr_to_png_bytes(result_bgr)
if png_bytes:
    st.download_button(
        "Unduh hasil",
        data=png_bytes,
        file_name=f"{os.path.splitext(st.session_state.active_name)[0]}_{view.lower()}.png",
        mime="image/png",
    )
