"""Brain Tumor MRI Segmentation with a custom U-Net (Streamlit).

Run:      streamlit run app.py
Weights:  best.pth in the same folder as this file.
"""

import base64
import html
import random
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Brain Tumor MRI Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _flat(s: str) -> str:
    """Collapse to one line so Markdown never splits an HTML block."""
    return " ".join(s.split())


# ──────────────────────────────────────────────────────────────────────────
# App styles (white background, black text)
# ──────────────────────────────────────────────────────────────────────────
APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--ink:#000;--muted:#4b5563;--line:#e2e5ea;--accent:#f26a1b;--soft:#f6f7f9;}
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"],[data-testid="stMain"]{
  background:#fff!important;color:#000;
  font-family:'Manrope',system-ui,-apple-system,'Segoe UI',sans-serif;}
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"],footer,#MainMenu{display:none!important;}
.block-container,[data-testid="stMainBlockContainer"]{
  padding:14px 28px 10px!important;max-width:1360px!important;}
[data-testid="stElementContainer"]:has(.hz),.element-container:has(.hz){display:none;}

/* top bar */
.topbar{display:flex;align-items:center;justify-content:space-between;height:48px;margin-bottom:10px;}
.brand{display:flex;align-items:center;gap:12px;}
.brand svg{width:34px;height:34px;flex:none;}
.brand-name{font-size:21px;font-weight:800;letter-spacing:-.015em;line-height:1.1;color:#000;}
.chips{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;}
.chip{font-size:12.5px;font-weight:600;padding:5px 11px;border:1px solid var(--line);
  border-radius:999px;background:#fff;color:#000;white-space:nowrap;}

/* panels */
[class*="st-key-panel_"]{border:1px solid var(--line)!important;border-radius:16px!important;
  background:#fff;padding:18px 20px!important;box-shadow:none!important;box-sizing:border-box;
  min-width:0;width:100%;}
[class*="st-key-panel_"] [data-testid="stVerticalBlock"]{width:100%;min-width:0;}
[class*="st-key-panel_"] [data-testid="stFileUploader"],
[class*="st-key-panel_"] [data-testid="stFileUploaderDropzone"]{width:100%;margin:0;box-sizing:border-box;}
[data-testid="stColumn"]{min-width:0;}
/* two main boxes: equal width, identical fixed-height rows, equal padding on all four sides */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [class*="st-key-panel_"]){align-items:flex-start;}
[class*="st-key-panel_"]{height:auto!important;overflow:visible!important;padding-bottom:20px!important;}
[class*="st-key-panel_"] [data-testid="stHorizontalBlock"]{margin:0;}
[class*="st-key-panel_"] [data-testid="stElementContainer"]:last-child,
[class*="st-key-panel_"] [data-testid="stHorizontalBlock"]:last-child{margin-bottom:0;}
.st-key-head_left,.st-key-head_right{height:40px;overflow:hidden;}
.st-key-ctrl_left,.st-key-ctrl_right{overflow:hidden;}
/* Streamlit gives markdown blocks a negative bottom margin; cancel it so padding is true on all sides */
[class*="st-key-panel_"] [data-testid="stMarkdown"],
[class*="st-key-panel_"] [data-testid="stMarkdownContainer"]{margin:0!important;}
[class*="st-key-panel_"] [data-testid="stVerticalBlock"]{gap:10px;}
.phead{display:flex;align-items:center;justify-content:space-between;gap:12px;min-height:40px;}
.ptitle{font-size:16px;font-weight:800;line-height:1.2;color:#000;}
.psub{font-size:13px;color:var(--muted);font-weight:500;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;max-width:44ch;}
.status{font-size:12.5px;font-weight:700;padding:5px 11px;border-radius:999px;white-space:nowrap;}
.status.wait{background:var(--soft);color:#374151;}
.status.hit{background:#fff1e6;color:#9a3a05;}
.status.clear{background:#e8f6ef;color:#0b6b46;}

/* uploader: compact dropzone, English copy regardless of browser language */
.st-key-ctrl_left,.st-key-ctrl_right{height:76px;display:flex;flex-direction:column;justify-content:center;}
[data-testid="stFileUploader"] label,[data-testid="stFileUploaderFile"],
[data-testid="stFileUploaderFileList"],[data-testid="stFileUploaderPagination"]{display:none!important;}
[data-testid="stFileUploaderDropzone"]{height:76px;box-sizing:border-box;border:1.5px dashed #b9c0cc;
  border-radius:12px;background:#fafbfc;padding:0 16px;display:flex;align-items:center;gap:12px;}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--accent);background:#fff8f3;}
[data-testid="stFileUploaderDropzoneInstructions"]{display:flex;flex-direction:column;gap:2px;}
[data-testid="stFileUploaderDropzoneInstructions"]>*{display:none!important;}
[data-testid="stFileUploaderDropzoneInstructions"]::before{content:"Drag and drop an MRI image here";
  font-size:14px;font-weight:700;color:#000;}
[data-testid="stFileUploaderDropzoneInstructions"]::after{content:"PNG, JPG or TIFF, up to 200 MB";
  font-size:12.5px;color:var(--muted);}
[data-testid="stFileUploaderDropzone"] button{display:none!important;}
[data-testid="stFileUploaderDeleteBtn"]{display:none!important;}
[data-testid="stFileUploaderDropzone"] > button{display:inline-flex!important;font-size:0!important;background:#000!important;
  border:1px solid #000!important;border-radius:9px!important;padding:7px 16px!important;}
[data-testid="stFileUploaderDropzone"] > button>*{display:none!important;}
[data-testid="stFileUploaderDropzone"] > button::after{content:"Browse files";font-size:13.5px;
  font-weight:700;color:#fff;}
[data-testid="stFileUploaderDropzone"] > button:hover{background:var(--accent)!important;border-color:var(--accent)!important;}

/* image stage */
.stage{height:clamp(220px,calc(100vh - 430px),740px);background:#050505;border-radius:12px;
  display:flex;align-items:center;justify-content:center;overflow:hidden;}
.stage img{max-width:100%;max-height:100%;object-fit:contain;display:block;}
.stage.empty{background:#fff;border:1.5px dashed var(--line);flex-direction:column;gap:8px;
  color:var(--muted);font-size:14px;font-weight:500;}
.stage.empty svg{width:34px;height:34px;}
.cap{text-align:center;font-size:13.5px;font-weight:700;color:#000;height:20px;line-height:20px;margin:0 0 6px;}

/* stats */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;}
.stat{min-width:0;border:1px solid var(--line);border-radius:10px;padding:8px 12px;background:#fff;
  height:58px;box-sizing:border-box;overflow:hidden;}
.stat .k{font-size:12px;color:var(--muted);font-weight:600;white-space:nowrap;}
.stat .v{font-size:18px;font-weight:800;letter-spacing:-.01em;line-height:1.3;color:#000;white-space:nowrap;}
[class*="st-key-panel_"] [data-testid="stHorizontalBlock"]{gap:10px;}

/* controls */
[data-testid="stDownloadButton"] button{width:auto;background:#000;border:1px solid #000;
  border-radius:10px;height:42px;padding:0 20px;}
[data-testid="stElementContainer"]:has([data-testid="stDownloadButton"]),
.element-container:has([data-testid="stDownloadButton"]){width:100%!important;display:flex;justify-content:flex-end;}
[data-testid="stDownloadButton"]{display:flex;justify-content:flex-end;margin-left:auto;}
[data-testid="stDownloadButton"] button p{color:#fff;font-weight:700;font-size:13.5px;white-space:nowrap;}
[data-testid="stDownloadButton"] button:hover{background:var(--accent);border-color:var(--accent);}
[data-testid="stDownloadButton"] button:disabled{background:#e5e7eb;border-color:#e5e7eb;}
[data-testid="stDownloadButton"] button:disabled p{color:#6b7280;}
.legend{height:76px;box-sizing:border-box;border:1px solid var(--line);border-radius:12px;background:var(--soft);
  display:flex;align-items:center;gap:12px;padding:0 16px;}
.legend .sw{width:18px;height:18px;border-radius:5px;background:#ff5a1f;flex:none;}
.legend .lt{font-size:14px;font-weight:700;color:#000;}
.legend .ls{font-size:12.5px;color:var(--muted);font-weight:500;}
.foot{text-align:left;font-size:12.5px;color:var(--muted);font-weight:500;margin-top:2px;}

@media (max-width:900px){
  .block-container,[data-testid="stMainBlockContainer"]{padding:12px 14px!important;}
  .stage{height:44vh;}
  .chips{display:none;}
}
"""

st.markdown(f"<style>{_flat(APP_CSS)}</style><span class='hz'></span>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Landing page (first load only): hold ~3.5 s, dissolve, app slides up
# ──────────────────────────────────────────────────────────────────────────
LANDING_CSS = """
.lp{position:fixed;inset:0;z-index:2147483000;display:flex;align-items:center;justify-content:center;
  overflow:hidden;background:radial-gradient(ellipse at 36% 50%,#0d2142 0%,#060d1f 55%,#030711 100%);
  font-family:'Manrope',system-ui,sans-serif;animation:lp-out 4.5s cubic-bezier(.4,0,.2,1) forwards;}
@keyframes lp-out{0%,78%{opacity:1;transform:none;filter:none;visibility:visible;}
  100%{opacity:0;transform:translateY(-5vh) scale(1.02);filter:blur(10px);visibility:hidden;}}
[data-testid="stElementContainer"]:has(.lp),.element-container:has(.lp){
  position:absolute;width:0;height:0;margin:0;overflow:visible;}
.block-container,[data-testid="stMainBlockContainer"]{position:relative;
  animation:app-up 1s cubic-bezier(.2,.8,.2,1) 3.5s both;}
@keyframes app-up{from{top:6vh;}to{top:0;}}
.lp-art{width:100%;height:100%;display:block;}
.lp-label{font:700 15px 'Manrope',system-ui,sans-serif;fill:#cfe3ff;}
.lp-fade{opacity:0;animation:lp-fade .9s ease-out forwards;}
@keyframes lp-fade{to{opacity:1;}}
.lp-trace{fill:none;stroke:#2b7de0;stroke-width:1.6;stroke-dasharray:1;stroke-dashoffset:1;
  animation:lp-draw 1.5s ease-out forwards;}
@keyframes lp-draw{to{stroke-dashoffset:0;}}
.lp-pulse{animation:lp-pulse 2.2s ease-in-out infinite;}
@keyframes lp-pulse{0%,100%{opacity:.55;}50%{opacity:1;}}
.lp-copy{position:absolute;left:6vw;bottom:7vh;color:#fff;opacity:0;animation:lp-fade 1s ease-out .9s forwards;}
.lp-title{font-size:clamp(22px,2.6vw,40px);font-weight:800;letter-spacing:-.02em;line-height:1.1;}
.lp-sub{font-size:clamp(13px,1.2vw,18px);color:#9fb6d8;margin-top:8px;font-weight:500;}
.lp-bar{position:absolute;left:0;bottom:0;height:3px;width:0;background:#ff7a1f;
  animation:lp-bar 3.5s linear forwards;}
@keyframes lp-bar{to{width:100%;}}
@media (prefers-reduced-motion:reduce){
  .lp-fade,.lp-trace,.lp-copy{animation:none!important;opacity:1;stroke-dashoffset:0;}
  .lp-pulse{animation:none!important;}
}
"""

BRAIN_D = (
    "M105 360 C70 300 95 215 165 175 C235 135 330 120 415 140 C505 160 575 225 580 305 "
    "C584 365 548 410 500 420 C470 425 450 440 430 450 C400 470 360 462 335 450 "
    "C300 480 235 490 195 460 C160 440 125 410 105 360 Z"
)
FRAME_Y = (54, 260, 466)  # top edge of each MRI frame (frames are 170 px tall, label sits above)


def _circuit_traces() -> str:
    """Blue circuit lines: edge -> brain (left) and brain -> MRI frames (right)."""
    rnd = random.Random(21)
    centers = [y + 85 for y in FRAME_Y]
    out = []
    # Left side: dim lines that run in from the edge toward the brain.
    for i in range(12):
        sy = 70 + i * 50 + rnd.randint(-10, 10)
        dy = rnd.randint(-60, 60)
        x1 = rnd.randint(30, 120)
        x2 = x1 + abs(dy)
        ex = x2 + rnd.randint(30, 110)
        ey = sy + dy
        out.append(
            f'<path class="lp-trace" pathLength="1" opacity=".5" style="animation-delay:{0.3 + i * 0.05:.2f}s" '
            f'd="M-10 {sy} H{x1} L{x2} {ey} H{ex}"/>'
            f'<circle class="lp-fade" cx="{ex}" cy="{ey}" r="3" fill="#4aa0ff" opacity=".6" style="animation-delay:1.4s"/>'
        )
    # Right side: 24 lines fan out to the three frames, ordered so they do not cross.
    for i in range(24):
        g, k = i // 8, i % 8
        sy = 232 + i * 9
        ty = centers[g] - 56 + k * 16
        x1 = rnd.randint(592, 620)
        x2 = x1 + abs(ty - sy)
        ex = rnd.randint(max(x2 + 10, 840), 874)
        out.append(
            f'<path class="lp-trace" pathLength="1" style="animation-delay:{0.5 + i * 0.04:.2f}s" '
            f'd="M480 {sy} H{x1} L{x2} {ty} H{ex}"/>'
            f'<circle class="lp-fade" cx="{ex}" cy="{ty}" r="3.2" fill="#6bb8ff" style="animation-delay:1.5s"/>'
        )
    return "".join(out)


def landing_html() -> str:
    labels = ("Original MRI", "Predicted tumor mask", "Original + tumor overlay")
    frames = []
    for i, y in enumerate(FRAME_Y):
        cy = y + 85
        place = f'transform="translate(1030 {cy}) scale(.8)"'
        if i == 0:
            inner = f'<use href="#lp-mri-g" {place}/>'
        elif i == 1:
            inner = (
                f'<rect x="932" y="{y + 2}" width="196" height="166" fill="#02060f"/>'
                f'<use href="#lp-tumor-shape" fill="#ff6a1a" {place}/>'
            )
        else:
            inner = (
                f'<use href="#lp-mri-g" {place}/>'
                f'<use href="#lp-tumor-shape" fill="#ff6a1a" fill-opacity=".5" stroke="#ff8a3d" stroke-width="2.5" {place}/>'
            )
        frames.append(
            f'<g class="lp-fade" style="animation-delay:{0.9 + i * 0.25:.2f}s">'
            f'<text class="lp-label" x="1030" y="{y - 8}" text-anchor="middle">{labels[i]}</text>'
            f'<rect x="930" y="{y}" width="200" height="170" rx="3" fill="#06142b" fill-opacity=".7" '
            f'stroke="#2f8bff" stroke-width="2"/>{inner}</g>'
            f'<g class="lp-fade" style="animation-delay:{1.5 + i * 0.15:.2f}s" fill="none" stroke="#ff8a1f" '
            f'stroke-width="3" stroke-linecap="round" stroke-linejoin="round">'
            f'<path d="M884 {cy} H922"/><path d="M911 {cy - 9} L924 {cy} L911 {cy + 9}"/></g>'
        )

    svg = f"""
<svg class="lp-art" viewBox="0 0 1200 700" preserveAspectRatio="xMidYMid meet">
<defs>
<linearGradient id="lp-fill" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#1d4380"/><stop offset=".5" stop-color="#112b58"/><stop offset="1" stop-color="#0a1c3b"/></linearGradient>
<linearGradient id="lp-line" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#6bb8ff"/><stop offset=".5" stop-color="#3d8bff"/><stop offset="1" stop-color="#6bb8ff"/></linearGradient>
<radialGradient id="lp-mri" cx=".5" cy=".5" r=".55"><stop offset="0" stop-color="#3b8cf0"/><stop offset=".75" stop-color="#14498f"/><stop offset="1" stop-color="#0a2a57"/></radialGradient>
<filter id="lp-fold" x="-10%" y="-10%" width="120%" height="120%"><feTurbulence type="fractalNoise" baseFrequency="0.028 0.04" numOctaves="2" seed="11" result="n"/><feDisplacementMap in="SourceGraphic" in2="n" scale="46" xChannelSelector="R" yChannelSelector="G"/></filter>
<filter id="lp-fold2" x="-10%" y="-10%" width="120%" height="120%"><feTurbulence type="fractalNoise" baseFrequency="0.05" numOctaves="2" seed="5" result="n"/><feDisplacementMap in="SourceGraphic" in2="n" scale="16" xChannelSelector="R" yChannelSelector="G"/></filter>
<filter id="lp-blur" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="8"/></filter>
<path id="lp-bp" d="{BRAIN_D}" vector-effect="non-scaling-stroke"/>
<clipPath id="lp-clip"><path d="{BRAIN_D}"/></clipPath>
<clipPath id="lp-mclip"><ellipse rx="72" ry="84"/></clipPath>
<path id="lp-tumor-shape" d="M20 -42 C38 -50 58 -38 56 -18 C54 0 40 10 24 6 C10 2 4 -14 10 -28 C12 -35 15 -40 20 -42 Z"/>
<g id="lp-mri-g">
<ellipse rx="82" ry="94" fill="#0a2550" stroke="#6db3ff" stroke-width="2.5"/>
<ellipse rx="72" ry="84" fill="url(#lp-mri)"/>
<g clip-path="url(#lp-mclip)"><g filter="url(#lp-fold2)" fill="none" stroke="#a9d4ff" stroke-width="1.4" opacity=".55"><ellipse rx="64" ry="76"/><ellipse rx="54" ry="66"/><ellipse rx="44" ry="56"/><ellipse rx="34" ry="46"/></g></g>
<path d="M-6 -34 C-18 -34 -25 -24 -23 -10 C-21 2 -15 10 -12 20 C-11 24 -8 23 -8 19 C-9 8 -12 -2 -11 -12 C-10 -20 -8 -27 -6 -34 Z" fill="#04122b" opacity=".9"/><path d="M6 -34 C18 -34 25 -24 23 -10 C21 2 15 10 12 20 C11 24 8 23 8 19 C9 8 12 -2 11 -12 C10 -20 8 -27 6 -34 Z" fill="#04122b" opacity=".9"/>
</g>
</defs>
{_circuit_traces()}
<g class="lp-fade" style="animation-delay:.15s">
<path d="M430 450 C440 410 520 410 548 440 C560 470 520 500 470 500 C440 495 425 475 430 450 Z" fill="#0e2447" stroke="#3d8bff" stroke-width="2"/>
<path d="M372 452 C372 500 380 540 392 572 L428 572 C424 530 426 490 430 455 Z" fill="#0b1c3b" stroke="#3d8bff" stroke-width="1.5"/>
<path d="M445 462 C470 452 510 452 535 462 M440 478 C470 468 515 468 540 478" fill="none" stroke="#6bb8ff" stroke-width="1.3" opacity=".6"/>
<path d="{BRAIN_D}" fill="none" stroke="url(#lp-line)" stroke-width="10" filter="url(#lp-blur)" opacity=".5"/>
<use href="#lp-bp" fill="url(#lp-fill)"/>
<g clip-path="url(#lp-clip)">
<g filter="url(#lp-fold)" fill="none" stroke="url(#lp-line)" stroke-width="2.6" stroke-linejoin="round" opacity=".7">
<use href="#lp-bp" transform="translate(335 310) scale(.93) translate(-335 -310)"/>
<use href="#lp-bp" transform="translate(335 310) scale(.82) translate(-335 -310)"/>
<use href="#lp-bp" transform="translate(335 310) scale(.71) translate(-335 -310)"/>
<use href="#lp-bp" transform="translate(335 310) scale(.60) translate(-335 -310)"/>
<use href="#lp-bp" transform="translate(335 310) scale(.49) translate(-335 -310)"/>
<use href="#lp-bp" transform="translate(335 310) scale(.38) translate(-335 -310)"/>
<use href="#lp-bp" transform="translate(335 310) scale(.27) translate(-335 -310)"/>
</g>
</g>
<path d="M205 425 C280 405 345 385 420 330" fill="none" stroke="#050c1c" stroke-width="5" stroke-linecap="round" opacity=".8"/>
<use href="#lp-bp" fill="none" stroke="url(#lp-line)" stroke-width="2.6"/>
</g>
{"".join(frames)}
</svg>
"""
    copy = (
        '<div class="lp-copy"><div class="lp-title">Brain Tumor MRI Segmentation</div>'
        '<div class="lp-sub">Upload a scan and see the predicted tumor region in seconds.</div></div>'
        '<div class="lp-bar"></div>'
    )
    return f"<style>{_flat(LANDING_CSS)}</style><div class=\"lp\" aria-hidden=\"true\">{_flat(svg)}{copy}</div>"


if not st.session_state.get("_landing_shown"):
    st.session_state["_landing_shown"] = True
    st.markdown(landing_html(), unsafe_allow_html=True)

# Heavy imports come after the landing page is sent, so it appears immediately.
import time  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# Model (architecture must match the custom U-Net that produced best.pth)
# ──────────────────────────────────────────────────────────────────────────
IMG_SIZE = 256
MODEL_PATH = Path(__file__).resolve().parent / "best.pth"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ACCENT_RGB = (255, 90, 31)
MIN_TUMOR_PIXELS = 25  # on the 256x256 grid; below this the result is treated as "no tumor found"


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 2, init_features: int = 16):
        super().__init__()
        f = init_features
        self.encoder1 = ConvBlock(in_channels, f)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ConvBlock(f, f * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = ConvBlock(f * 2, f * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = ConvBlock(f * 4, f * 8)
        self.pool4 = nn.MaxPool2d(2)
        self.encoder5 = ConvBlock(f * 8, f * 16)

        self.upconv4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(f * 16, f * 8)
        self.upconv3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(f * 8, f * 4)
        self.upconv2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(f * 4, f * 2)
        self.upconv1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(f * 2, f)
        self.output = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bottleneck = self.encoder5(self.pool4(enc4))
        x = self.decoder4(torch.cat((self.upconv4(bottleneck), enc4), dim=1))
        x = self.decoder3(torch.cat((self.upconv3(x), enc3), dim=1))
        x = self.decoder2(torch.cat((self.upconv2(x), enc2), dim=1))
        x = self.decoder1(torch.cat((self.upconv1(x), enc1), dim=1))
        return self.output(x)


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, out_channels=2, init_features=16)
    # Only load weight files from sources you trust.
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, device


def preprocess_image(rgb_image: np.ndarray) -> torch.Tensor:
    """Match training: RGB -> resize -> /255 -> ImageNet normalize -> CHW."""
    resized = cv2.resize(rgb_image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    image = resized.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    image = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0)


@st.cache_data(show_spinner=False, max_entries=6)
def segment(file_bytes: bytes):
    """Run the model once per uploaded file; returns images and stats, or None if unreadable."""
    bgr = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    model, device = load_model()
    x = preprocess_image(rgb).to(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        probs = torch.softmax(model(x), dim=1)[0]
    mask_256 = probs.argmax(dim=0).cpu().numpy().astype(np.uint8)
    tumor_prob = probs[1].cpu().numpy()
    infer_ms = (time.perf_counter() - t0) * 1000

    # Nearest-neighbor keeps the class IDs binary when scaling back to the original size.
    mask = cv2.resize(mask_256, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = rgb.copy()
    fill = np.zeros_like(rgb)
    fill[:] = ACCENT_RGB
    blended = cv2.addWeighted(rgb, 0.62, fill, 0.38, 0)
    overlay[mask == 1] = blended[mask == 1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, contours, -1, ACCENT_RGB, max(1, round(max(h, w) / 300)))

    mask_view = np.zeros_like(rgb)
    mask_view[mask == 1] = ACCENT_RGB

    tumor_px = int(np.count_nonzero(mask_256))
    ok, png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return {
        "rgb": rgb,
        "overlay": overlay,
        "mask_view": mask_view,
        "overlay_png": png.tobytes() if ok else b"",
        "width": w,
        "height": h,
        "area_pct": 100.0 * tumor_px / mask_256.size,
        "confidence": float(tumor_prob[mask_256 == 1].mean()) * 100 if tumor_px else None,
        "infer_ms": infer_ms,
        "detected": tumor_px >= MIN_TUMOR_PIXELS,
        "tumor_px": tumor_px,
    }


# ──────────────────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────────────────
def data_uri(img: np.ndarray, ext: str = ".jpg", max_side: int = 1000) -> str:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(ext, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buf).decode()}"


def stage(img: np.ndarray | None, alt: str, empty_text: str, caption: str, ext: str = ".jpg") -> str:
    if img is None:
        icon = (
            '<svg viewBox="0 0 24 24" fill="none" stroke="#9aa3af" stroke-width="1.6" stroke-linecap="round" '
            'stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="3"/>'
            '<circle cx="9" cy="10" r="1.8"/><path d="M4 18l5-5 4 4 3-3 4 4"/></svg>'
        )
        box = f'<div class="stage empty">{icon}<span>{empty_text}</span></div>'
    else:
        box = f'<div class="stage"><img src="{data_uri(img, ext)}" alt="{html.escape(alt)}"></div>'
    return f'<div class="cap">{caption}</div>{box}'


def stats(items: list[tuple[str, str]]) -> str:
    cells = "".join(f'<div class="stat"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in items)
    return f'<div class="stats">{cells}</div>'


def size_label(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 * 1024 else f"{n / 1024 / 1024:.1f} MB"


# ──────────────────────────────────────────────────────────────────────────
# Page
# ──────────────────────────────────────────────────────────────────────────
try:
    _, device = load_model()
except FileNotFoundError:
    st.error(f"Model weights not found: {MODEL_PATH.name}. Place the trained best.pth next to app.py.")
    st.stop()
except (RuntimeError, ValueError, TypeError) as exc:
    st.error("The weights could not be loaded. Check that best.pth is a state_dict from the custom U-Net (init_features=16).")
    st.code(str(exc))
    st.stop()

st.markdown(
    _flat(
        f"""
<div class="topbar">
  <div class="brand">
    <svg viewBox="0 0 24 24" fill="none" stroke="#000" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
    <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>
    <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>
    <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4" stroke="#f26a1b"/>
    <path d="M17.599 6.5a3 3 0 0 0 .399-1.375"/><path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"/>
    <path d="M3.477 10.896a4 4 0 0 1 .585-.396"/><path d="M19.938 10.5a4 4 0 0 1 .585.396"/>
    <path d="M6 18a4 4 0 0 1-1.967-.516"/><path d="M19.967 17.484A4 4 0 0 1 18 18"/></svg>
    <div class="brand-name">Brain Tumor MRI Segmentation</div>
  </div>
  <div class="chips">
    <span class="chip">Custom U-Net</span>
    <span class="chip">Input {IMG_SIZE} × {IMG_SIZE}</span>
    <span class="chip">Running on {str(device).upper()}</span>
  </div>
</div>
"""
    ),
    unsafe_allow_html=True,
)

if "uploader_n" not in st.session_state:
    st.session_state["uploader_n"] = 0

col_left, col_right = st.columns(2, gap="medium")
with col_left:
    panel_l = st.container(border=True, key="panel_left")
with col_right:
    panel_r = st.container(border=True, key="panel_right")

# Left: header slot, uploader
with panel_l:
    head_l = st.container(key="head_left")
    with st.container(key="ctrl_left"):
        uploaded = st.file_uploader(
            "MRI image",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            key=f"mri_{st.session_state['uploader_n']}",
            label_visibility="collapsed",
        )

file_bytes = uploaded.getvalue() if uploaded is not None else None
result = segment(file_bytes) if file_bytes else None

# Left: header, preview, file info
with panel_l:
    with head_l:
        h1, h2 = st.columns([5, 1], vertical_alignment="center")
        with h1:
            sub = html.escape(uploaded.name) if uploaded is not None else "Upload a brain MRI to begin"
            st.markdown(
                f'<div class="phead"><div><div class="ptitle">MRI scan</div><div class="psub">{sub}</div></div></div>',
                unsafe_allow_html=True,
            )
        with h2:
            if uploaded is not None and st.button("Clear", key="clear", type="tertiary"):
                st.session_state["uploader_n"] += 1
                st.rerun()
    if file_bytes and result is None:
        st.error("This file could not be read. Upload a valid 8-bit PNG, JPG or TIFF image.")
    st.markdown(
        stage(result["rgb"] if result else None, "Uploaded MRI scan", "Your scan will appear here", "Original MRI"),
        unsafe_allow_html=True,
    )
    fmt = Path(uploaded.name).suffix.lstrip(".").upper() if uploaded is not None else "—"
    st.markdown(
        stats(
            [
                ("Dimensions", f"{result['width']} × {result['height']}" if result else "—"),
                ("File size", size_label(len(file_bytes)) if file_bytes else "—"),
                ("Format", fmt or "—"),
                ("Color", "RGB" if result else "—"),
            ]
        ),
        unsafe_allow_html=True,
    )

# Right: segmentation
with panel_r:
    if result is None:
        status = '<span class="status wait">Waiting for scan</span>'
    elif result["detected"]:
        status = '<span class="status hit">Tumor region found</span>'
    else:
        status = '<span class="status clear">No tumor region found</span>'
    with st.container(key="head_right"):
        st.markdown(
            f'<div class="phead"><div><div class="ptitle">Segmentation</div>'
            f'<div class="psub">Predicted tumor region</div></div>{status}</div>',
            unsafe_allow_html=True,
        )
    with st.container(key="ctrl_right"):
        st.markdown(
            '<div class="legend"><span class="sw"></span><div>'
            '<div class="lt">Orange marks the predicted tumor region</div>'
            '<div class="ls">Everything else is classified as background</div></div></div>',
            unsafe_allow_html=True,
        )
    mask_col, overlay_col = st.columns(2)
    with mask_col:
        st.markdown(
            stage(
                result["mask_view"] if result else None,
                "Predicted tumor mask",
                "Mask appears here",
                "Predicted tumor mask",
                ".png",
            ),
            unsafe_allow_html=True,
        )
    with overlay_col:
        st.markdown(
            stage(
                result["overlay"] if result else None,
                "MRI with tumor overlay",
                "Overlay appears here",
                "Original + tumor overlay",
            ),
            unsafe_allow_html=True,
        )
    st.markdown(
        stats(
            [
                ("Tumor area", f"{result['area_pct']:.2f}%" if result else "—"),
                (
                    "Mean confidence",
                    f"{result['confidence']:.1f}%" if result and result["confidence"] is not None else "—",
                ),
                ("Tumor pixels", f"{result['tumor_px']:,}" if result else "—"),
                ("Inference time", f"{result['infer_ms']:.0f} ms" if result else "—"),
            ]
        ),
        unsafe_allow_html=True,
    )

# Bottom row, outside the boxes: disclaimer (left) and download (right, aligned with the right box)
foot_l, foot_r = st.columns(2, gap="medium", vertical_alignment="center")
with foot_l:
    st.markdown(
        '<div class="foot">For research and education only. Not a medical device and not for clinical diagnosis.</div>',
        unsafe_allow_html=True,
    )
with foot_r:
    stem = Path(uploaded.name).stem if uploaded is not None else "scan"
    st.download_button(
        "Download overlay",
        data=result["overlay_png"] if result else b"",
        file_name=f"{stem}_segmentation.png",
        mime="image/png",
        disabled=result is None,
    )
