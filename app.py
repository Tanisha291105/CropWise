import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="CropWise - Crop Recommendation",
    page_icon="🌱",
    layout="wide"
)


# ==================================================
# 2. FILE PATHS
# ==================================================

MODEL_PATH = "model/crop_model.pkl"
DATASET_PATH = "dataset/Crop_recommendation.csv"


# ==================================================
# 3. CUSTOM CSS & MODERN ANIMATIONS
# ==================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
    --primary: #10b981;
    --primary-hover: #059669;
    --primary-dark: #047857;
    --primary-light: #34d399;
    --mint-glow: rgba(16, 185, 129, 0.16);
    --forest-900: #051d13;
    --forest-800: #092c1e;
    --forest-700: #0f422e;
    --forest-600: #1a563d;
    --paper: #f7faf7;
    --paper-card: rgba(255, 255, 255, 0.92);
    --ink: #12281d;
    --ink-muted: #536e60;
    --border-subtle: rgba(16, 185, 129, 0.14);
    --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-display: 'Outfit', 'Plus Jakarta Sans', sans-serif;
}

/* Global resets & typography */
html, body, [class*="css"], .stApp {
    font-family: var(--font-sans);
    letter-spacing: -0.012em;
}

.stApp,
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 12% 15%, rgba(16, 185, 129, 0.06) 0%, transparent 45%),
                radial-gradient(circle at 88% 82%, rgba(52, 211, 153, 0.07) 0%, transparent 50%),
                linear-gradient(180deg, #f8faf8 0%, #f1f6f2 100%);
    color: var(--ink);
}

[data-testid="stAppViewContainer"] {
    isolation: isolate;
    position: relative;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #eef3ee;
}
::-webkit-scrollbar-thumb {
    background: #b5cebc;
    border-radius: 99px;
}
::-webkit-scrollbar-thumb:hover {
    background: #87aa91;
}

/* Ambient Floating Particles & Botanical Leaves */
.ambient-leaves {
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    position: fixed;
    z-index: 0;
}

.ambient-leaf {
    border-radius: 100% 0 100% 0;
    bottom: -3.5rem;
    height: 22px;
    left: var(--left);
    opacity: 0;
    position: absolute;
    transform: rotate(35deg);
    width: 14px;
    will-change: transform;
    filter: drop-shadow(0 3px 6px rgba(16, 185, 129, 0.3));
}

/* Leaf Variations for Organic Diversity */
.leaf-type-emerald {
    background: linear-gradient(
        135deg,
        rgba(34, 197, 94, 0.92) 0%,
        rgba(16, 185, 129, 0.88) 55%,
        rgba(5, 150, 105, 0.94) 100%
    );
    border: 1px solid rgba(52, 211, 153, 0.5);
}
.leaf-type-emerald::after {
    background: rgba(4, 95, 59, 0.85);
    content: "";
    height: 72%;
    left: 50%;
    position: absolute;
    top: 14%;
    transform: rotate(10deg);
    width: 1.5px;
}

.leaf-type-mint {
    background: linear-gradient(
        135deg,
        rgba(74, 222, 128, 0.92) 0%,
        rgba(52, 211, 153, 0.88) 50%,
        rgba(16, 185, 129, 0.88) 100%
    );
    border: 1px solid rgba(110, 231, 183, 0.55);
    border-radius: 0 100% 0 100%;
}
.leaf-type-mint::after {
    background: rgba(5, 150, 105, 0.85);
    content: "";
    height: 72%;
    left: 50%;
    position: absolute;
    top: 14%;
    transform: rotate(-10deg);
    width: 1.5px;
}

.leaf-type-forest {
    background: linear-gradient(
        135deg,
        rgba(16, 185, 129, 0.92) 0%,
        rgba(5, 150, 105, 0.9) 55%,
        rgba(4, 120, 87, 0.95) 100%
    );
    border: 1px solid rgba(16, 185, 129, 0.45);
}
.leaf-type-forest::after {
    background: rgba(2, 60, 37, 0.9);
    content: "";
    height: 72%;
    left: 50%;
    position: absolute;
    top: 14%;
    transform: rotate(12deg);
    width: 1.5px;
}

/* Animations & Trajectories */
.anim-sway-right {
    animation: leaf-drift-right var(--duration) linear var(--delay) infinite;
}

.anim-sway-left {
    animation: leaf-drift-left var(--duration) linear var(--delay) infinite;
}

.anim-flutter {
    animation: leaf-drift-flutter var(--duration) ease-in-out var(--delay) infinite;
}

@keyframes leaf-drift-right {
    0% {
        opacity: 0;
        transform: translate3d(0, 0, 0) rotate(15deg) scale(var(--scale, 1));
    }
    10% {
        opacity: 0.88;
    }
    50% {
        opacity: 0.82;
        transform: translate3d(45px, -52vh, 0) rotate(195deg) scale(var(--scale, 1));
    }
    85% {
        opacity: 0.86;
    }
    100% {
        opacity: 0;
        transform: translate3d(-20px, -114vh, 0) rotate(375deg) scale(var(--scale, 1));
    }
}

@keyframes leaf-drift-left {
    0% {
        opacity: 0;
        transform: translate3d(0, 0, 0) rotate(-20deg) scale(var(--scale, 1));
    }
    10% {
        opacity: 0.88;
    }
    50% {
        opacity: 0.82;
        transform: translate3d(-50px, -54vh, 0) rotate(165deg) scale(var(--scale, 1));
    }
    85% {
        opacity: 0.86;
    }
    100% {
        opacity: 0;
        transform: translate3d(25px, -114vh, 0) rotate(345deg) scale(var(--scale, 1));
    }
}

@keyframes leaf-drift-flutter {
    0% {
        opacity: 0;
        transform: translate3d(0, 0, 0) rotate(25deg) scale(var(--scale, 1));
    }
    12% {
        opacity: 0.9;
    }
    30% {
        transform: translate3d(32px, -32vh, 0) rotate(110deg) scale(var(--scale, 1));
    }
    60% {
        transform: translate3d(-35px, -68vh, 0) rotate(240deg) scale(var(--scale, 1));
    }
    85% {
        opacity: 0.88;
    }
    100% {
        opacity: 0;
        transform: translate3d(15px, -114vh, 0) rotate(390deg) scale(var(--scale, 1));
    }
}

/* Individual Leaf Positions & Timings (Balanced Set) */
.leaf-1  { --delay: -2s;  --duration: 24s; --left: 5%;  --scale: 1.05; }
.leaf-2  { --delay: -14s; --duration: 29s; --left: 15%; --scale: 0.85; }
.leaf-3  { --delay: -7s;  --duration: 25s; --left: 25%; --scale: 1.15; }
.leaf-4  { --delay: -19s; --duration: 31s; --left: 37%; --scale: 0.9;  }
.leaf-5  { --delay: -4s;  --duration: 23s; --left: 48%; --scale: 1.1;  }
.leaf-6  { --delay: -16s; --duration: 28s; --left: 59%; --scale: 0.85; }
.leaf-7  { --delay: -9s;  --duration: 26s; --left: 70%; --scale: 1.2;  }
.leaf-8  { --delay: -22s; --duration: 32s; --left: 80%; --scale: 0.95; }
.leaf-9  { --delay: -11s; --duration: 27s; --left: 89%; --scale: 1.1;  }
.leaf-10 { --delay: -18s; --duration: 30s; --left: 96%; --scale: 0.85; }

/* Ambient Glowing Spores */
.ambient-spore {
    position: absolute;
    bottom: -2.5rem;
    left: var(--left);
    width: var(--size, 7px);
    height: var(--size, 7px);
    border-radius: 50%;
    background: radial-gradient(circle, rgba(52, 211, 153, 0.9) 0%, rgba(16, 185, 129, 0.4) 60%, transparent 100%);
    box-shadow: 0 0 10px rgba(52, 211, 153, 0.7);
    opacity: 0;
    animation: spore-drift var(--duration, 20s) ease-in-out var(--delay, 0s) infinite;
    pointer-events: none;
}

.spore-one   { --left: 18%; --size: 7px; --delay: -3s;  --duration: 19s; }
.spore-two   { --left: 52%; --size: 8px; --delay: -12s; --duration: 23s; }
.spore-three { --left: 84%; --size: 6px; --delay: -7s;  --duration: 21s; }

@keyframes spore-drift {
    0% {
        opacity: 0;
        transform: translate3d(0, 0, 0) scale(0.6);
    }
    20% {
        opacity: 0.8;
    }
    60% {
        opacity: 0.7;
        transform: translate3d(30px, -52vh, 0) scale(1.15);
    }
    85% {
        opacity: 0.8;
    }
    100% {
        opacity: 0;
        transform: translate3d(-20px, -112vh, 0) scale(0.7);
    }
}

[data-testid="stHeader"] {
    background: transparent;
}

.main .block-container {
    max-width: 1140px;
    padding: 2rem 2.25rem 3.5rem;
}

/* Sidebar Modernization */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #051d13 0%, #092c1e 55%, #051e14 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.14);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    color: #f1f8f3;
}

.sidebar-brand {
    font-family: var(--font-display);
    color: #ffffff;
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    padding: 0.5rem 0 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.55rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.09);
    margin-bottom: 1.25rem;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    color: #a1c2ac !important;
    font-family: var(--font-display);
    font-size: 0.76rem !important;
    font-weight: 750 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.6rem !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 14px;
    padding: 0.65rem 0.95rem;
    margin-bottom: 0.55rem;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.09);
    border-color: rgba(52, 211, 153, 0.35);
    transform: translateX(3px);
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.22) 0%, rgba(5, 150, 105, 0.12) 100%);
    border-color: #10b981;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.25);
}

[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"] > label p {
    color: #f3f8f4 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}

/* Trendy Hero Section */
.hero {
    position: relative;
    background: linear-gradient(135deg, #051d13 0%, #093321 42%, #125537 80%, #082d1e 100%);
    background-size: 200% 200%;
    animation: heroMesh 14s ease infinite;
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 28px;
    color: #f7faf6;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 2rem;
    min-height: 205px;
    overflow: hidden;
    padding: 2.25rem 2.85rem;
    box-shadow: 0 20px 42px -14px rgba(5, 29, 19, 0.42),
                0 0 0 1px rgba(255, 255, 255, 0.1) inset;
    backdrop-filter: blur(12px);
}

@keyframes heroMesh {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero-glow-orb {
    position: absolute;
    top: -65px;
    right: 18%;
    width: 270px;
    height: 270px;
    background: radial-gradient(circle, rgba(52, 211, 153, 0.26) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    animation: orbPulse 7s ease-in-out infinite alternate;
}

@keyframes orbPulse {
    0% { transform: scale(1) translate(0, 0); opacity: 0.6; }
    100% { transform: scale(1.3) translate(-25px, 20px); opacity: 0.92; }
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    background: rgba(255, 255, 255, 0.11);
    border: 1px solid rgba(255, 255, 255, 0.18);
    padding: 0.35rem 0.85rem;
    border-radius: 99px;
    margin-bottom: 0.8rem;
    backdrop-filter: blur(8px);
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background: #34d399;
    border-radius: 50%;
    box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7);
    animation: radarPulse 2s infinite;
}

@keyframes radarPulse {
    0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.7); }
    70% { box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }
    100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
}

.hero-eyebrow {
    color: #a7f3d0;
    font-family: var(--font-display);
    font-size: 0.74rem;
    font-weight: 750;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

.hero h1 {
    font-family: var(--font-display);
    color: #ffffff;
    font-size: clamp(2.1rem, 4.3vw, 3.2rem);
    font-weight: 800;
    letter-spacing: -0.05em;
    line-height: 1.1;
    margin: 0.35rem 0 0.65rem;
    text-shadow: 0 2px 12px rgba(0, 0, 0, 0.2);
}

.hero p {
    color: #d1fae5;
    font-size: 1.02rem;
    line-height: 1.62;
    margin: 0;
    max-width: 590px;
    opacity: 0.94;
}

.hero-mark-wrap {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-left: 2rem;
    z-index: 2;
}

.hero-mark-ring {
    position: absolute;
    inset: -10px;
    border-radius: 34px;
    border: 2px dashed rgba(52, 211, 153, 0.35);
    animation: ringRotate 24s linear infinite;
}

@keyframes ringRotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.hero-mark {
    align-items: center;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.2) 0%, rgba(255, 255, 255, 0.07) 100%);
    border: 1px solid rgba(255, 255, 255, 0.26);
    border-radius: 28px;
    display: flex;
    font-size: 3.5rem;
    height: 104px;
    justify-content: center;
    min-width: 104px;
    backdrop-filter: blur(14px);
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.2);
    animation: floatMark 4s ease-in-out infinite;
}

@keyframes floatMark {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-7px); }
}

/* Section titles & Headings */
.section-title {
    font-family: var(--font-display);
    color: var(--forest-900);
    font-size: 1.55rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    margin-bottom: 0.25rem;
}

.section-heading {
    align-items: center;
    color: var(--forest-900);
    display: flex;
    font-family: var(--font-display);
    font-size: 1.15rem;
    font-weight: 750;
    gap: 0.75rem;
    margin-bottom: 1.1rem;
}

.step-number {
    align-items: center;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%);
    border-radius: 10px;
    color: #ffffff;
    display: inline-flex;
    font-family: var(--font-sans);
    font-size: 0.82rem;
    font-weight: 750;
    height: 30px;
    justify-content: center;
    width: 30px;
    box-shadow: 0 3px 10px rgba(16, 185, 129, 0.35);
    transition: transform 0.25s ease;
}

/* Glassmorphic Container Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--paper-card);
    border: 1px solid rgba(16, 185, 129, 0.15) !important;
    border-radius: 22px !important;
    box-shadow: 0 4px 20px -2px rgba(10, 40, 26, 0.05),
                0 0 0 1px rgba(255, 255, 255, 0.8) inset !important;
    backdrop-filter: blur(14px);
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    margin-bottom: 1.25rem;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(16, 185, 129, 0.32) !important;
    box-shadow: 0 14px 34px -4px rgba(16, 185, 129, 0.12),
                0 0 0 1px rgba(16, 185, 129, 0.15) inset !important;
    transform: translateY(-2px);
}

[data-testid="stVerticalBlockBorderWrapper"]:hover .step-number {
    transform: scale(1.08) rotate(3deg);
}

/* Input Fields */
[data-testid="stNumberInput"] {
    margin-bottom: 0.5rem;
}

[data-testid="stNumberInput"] label {
    color: #244132 !important;
    font-size: 0.88rem !important;
    font-weight: 650 !important;
    letter-spacing: -0.01em;
    margin-bottom: 0.35rem;
}

[data-testid="stNumberInput"] div[data-baseweb="input"] {
    background: #ffffff !important;
    border: 1.5px solid #dce8dd !important;
    border-radius: 13px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) inset;
}

[data-testid="stNumberInput"] div[data-baseweb="input"]:hover {
    border-color: #a4cead !important;
}

[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within {
    border-color: #10b981 !important;
    box-shadow: 0 0 0 3.5px rgba(16, 185, 129, 0.18) !important;
    background: #fcfffc !important;
}

[data-testid="stNumberInput"] input {
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    color: #122b1e !important;
    -webkit-text-fill-color: #122b1e !important;
    font-size: 0.96rem !important;
    padding: 0.6rem 0.85rem !important;
}

[data-testid="stNumberInput"] button {
    border-radius: 8px !important;
    color: #2f6946 !important;
    transition: background 0.2s ease !important;
}

[data-testid="stNumberInput"] button:hover {
    background: #e8f5ec !important;
    color: #059669 !important;
}

/* Button Animations & Styling */
div.stButton > button {
    font-family: var(--font-sans);
    font-weight: 700;
    font-size: 0.95rem;
    border-radius: 14px;
    min-height: 3.1rem;
    letter-spacing: -0.01em;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    cursor: pointer;
}

/* Primary "Get my crop recommendation" Button */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #059669 0%, #10b981 50%, #047857 100%) !important;
    background-size: 200% auto;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    box-shadow: 0 8px 24px -4px rgba(16, 185, 129, 0.42),
                0 2px 6px -1px rgba(16, 185, 129, 0.2) !important;
    color: #ffffff !important;
    width: 100%;
    font-size: 1.05rem;
    font-weight: 750;
    padding: 0.85rem 1.8rem;
}

div.stButton > button[kind="primary"]::before,
div.stButton > button[data-testid="baseButton-primary"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: -100%;
    width: 70%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.28), transparent);
    transform: skewX(-25deg);
    animation: btnShine 4.5s infinite;
}

@keyframes btnShine {
    0%, 75% { left: -100%; }
    100% { left: 200%; }
}

div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="baseButton-primary"]:hover {
    background-position: right center !important;
    transform: translateY(-2px);
    box-shadow: 0 14px 30px -4px rgba(16, 185, 129, 0.55),
                0 4px 10px rgba(0, 0, 0, 0.08) !important;
    color: #ffffff !important;
}

div.stButton > button[kind="primary"]:active,
div.stButton > button[data-testid="baseButton-primary"]:active {
    transform: translateY(1px);
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
}

/* Secondary Buttons ("Try Sample Values") */
div.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {
    background: #ffffff !important;
    border: 1.5px solid #d4e4d6 !important;
    color: #124028 !important;
    box-shadow: 0 2px 8px rgba(10, 40, 26, 0.04) !important;
    border-radius: 99px !important;
    padding: 0.45rem 1.4rem !important;
    min-height: 2.6rem !important;
}

div.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):hover {
    background: #f0fdf4 !important;
    border-color: #10b981 !important;
    color: #059669 !important;
    transform: translateY(-1.5px);
    box-shadow: 0 6px 18px rgba(16, 185, 129, 0.16) !important;
}

/* Result Card & Entrance Animation */
.result-card {
    position: relative;
    background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 45%, #f7fee7 100%);
    border: 1.5px solid rgba(16, 185, 129, 0.32);
    border-radius: 26px;
    margin: 1.6rem 0 1.8rem;
    padding: 2.2rem 2.4rem;
    box-shadow: 0 20px 45px -10px rgba(16, 185, 129, 0.2),
                0 0 0 1px rgba(255, 255, 255, 0.95) inset;
    overflow: hidden;
    animation: resultReveal 0.65s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

@keyframes resultReveal {
    0% {
        opacity: 0;
        transform: translateY(22px) scale(0.97);
    }
    100% {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

.result-card::before {
    content: "";
    position: absolute;
    top: -50%;
    right: -20%;
    width: 250px;
    height: 250px;
    background: radial-gradient(circle, rgba(52, 211, 153, 0.22) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
    animation: orbPulse 6s ease-in-out infinite alternate;
}

.result-badge-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 0.75rem;
    margin-bottom: 0.9rem;
    position: relative;
    z-index: 2;
}

.result-eyebrow {
    color: #059669;
    font-family: var(--font-display);
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(16, 185, 129, 0.12);
    padding: 0.3rem 0.8rem;
    border-radius: 99px;
    border: 1px solid rgba(16, 185, 129, 0.2);
}

.confidence-pill {
    font-family: var(--font-display);
    font-size: 0.8rem;
    font-weight: 750;
    color: #065f46;
    background: #d1fae5;
    border: 1px solid #a7f3d0;
    padding: 0.3rem 0.85rem;
    border-radius: 99px;
    box-shadow: 0 2px 6px rgba(16, 185, 129, 0.12);
}

.crop-header-wrap {
    display: flex;
    align-items: center;
    gap: 1.1rem;
    margin: 0.4rem 0 0.8rem;
    position: relative;
    z-index: 2;
}

.crop-emoji-animated {
    font-size: clamp(3rem, 5.5vw, 4.2rem);
    line-height: 1;
    filter: drop-shadow(0 8px 16px rgba(16, 185, 129, 0.25));
    animation: bounceCrop 3s ease-in-out infinite;
    display: inline-block;
}

@keyframes bounceCrop {
    0%, 100% { transform: translateY(0) rotate(0deg); }
    35% { transform: translateY(-7px) rotate(-4deg); }
    70% { transform: translateY(-2px) rotate(3deg); }
}

.crop-name {
    font-family: var(--font-display);
    font-size: clamp(2.3rem, 4.5vw, 3.2rem);
    font-weight: 800;
    letter-spacing: -0.05em;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #064e3b 0%, #047857 55%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.result-description {
    color: #3b5647;
    font-size: 1.05rem;
    line-height: 1.65;
    margin: 0.6rem 0 0.2rem;
    max-width: 720px;
    position: relative;
    z-index: 2;
}

/* Expanders & Metrics & History */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid #e1ede2 !important;
    border-radius: 18px !important;
    box-shadow: 0 2px 10px rgba(10, 40, 26, 0.03) !important;
    overflow: hidden;
    margin-bottom: 1rem !important;
    transition: border-color 0.25s ease;
}

[data-testid="stExpander"]:hover {
    border-color: #b3d9b8 !important;
}

[data-testid="stExpander"] summary {
    font-family: var(--font-sans) !important;
    font-weight: 650 !important;
    color: #193828 !important;
    padding: 0.85rem 1.15rem !important;
}

[data-testid="stExpander"] summary:hover {
    color: #059669 !important;
}

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e0ebe2;
    border-radius: 18px;
    padding: 1.25rem 1.6rem;
    box-shadow: 0 4px 16px rgba(10, 40, 26, 0.04);
    display: inline-block;
    min-width: 220px;
    margin-bottom: 1.25rem;
    position: relative;
    overflow: hidden;
}

[data-testid="stMetric"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, #10b981 0%, #059669 100%);
}

[data-testid="stMetricLabel"] p {
    font-family: var(--font-display);
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    color: #557262 !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

[data-testid="stMetricValue"] div {
    font-family: var(--font-display) !important;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    color: #064e3b !important;
    letter-spacing: -0.04em;
}

/* Alert styling */
[data-testid="stAlert"] {
    border-radius: 16px !important;
    border: 1px solid #d5e8d8 !important;
    background: #f4faf5 !important;
}

/* Footer note */
.footer-note {
    font-family: var(--font-sans);
    color: #637f70;
    font-size: 0.88rem;
    font-weight: 500;
    padding: 1.2rem 0;
    text-align: center;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
}

/* Responsive */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1.2rem 1rem 2.5rem;
    }

    .hero {
        flex-direction: column;
        align-items: flex-start;
        padding: 1.6rem 1.8rem;
        gap: 1.4rem;
        border-radius: 20px;
    }

    .hero-mark-wrap {
        margin-left: 0;
        align-self: flex-end;
    }

    .result-badge-row {
        flex-direction: column;
        align-items: flex-start;
    }
}

@media (prefers-reduced-motion: reduce) {
    *, ::before, ::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
    .ambient-leaves { display: none; }
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="ambient-leaves" aria-hidden="true">
        <span class="ambient-leaf leaf-type-emerald anim-sway-right leaf-1"></span>
        <span class="ambient-leaf leaf-type-mint anim-sway-left leaf-2"></span>
        <span class="ambient-leaf leaf-type-forest anim-flutter leaf-3"></span>
        <span class="ambient-leaf leaf-type-emerald anim-sway-right leaf-4"></span>
        <span class="ambient-leaf leaf-type-mint anim-flutter leaf-5"></span>
        <span class="ambient-leaf leaf-type-forest anim-sway-left leaf-6"></span>
        <span class="ambient-leaf leaf-type-emerald anim-sway-right leaf-7"></span>
        <span class="ambient-leaf leaf-type-mint anim-flutter leaf-8"></span>
        <span class="ambient-leaf leaf-type-forest anim-sway-left leaf-9"></span>
        <span class="ambient-leaf leaf-type-emerald anim-sway-right leaf-10"></span>
        <span class="ambient-spore spore-one"></span>
        <span class="ambient-spore spore-two"></span>
        <span class="ambient-spore spore-three"></span>
    </div>
    """,
    unsafe_allow_html=True
)


# ==================================================
# 4. LOAD TRAINED MODEL
# ==================================================

if not os.path.exists(MODEL_PATH):

    st.error("Trained model not found!")

    st.info("Please check model/crop_model.pkl")

    st.stop()

try:

    model = joblib.load(MODEL_PATH)

except Exception as error:

    st.error("Unable to load the trained model.")

    st.code(str(error))

    st.stop()


# ==================================================
# 5. LOAD DATASET
# ==================================================

dataset = None

if os.path.exists(DATASET_PATH):

    try:
        dataset = pd.read_csv(DATASET_PATH)

    except Exception:
        st.warning("Dataset could not be loaded.")


# ==================================================
# 6. CROP INFORMATION
# ==================================================

crop_info = {

    "rice": "A cereal crop commonly grown in warm and water-rich conditions.",

    "maize": "A cereal crop used for food, animal feed and industrial products.",

    "chickpea": "A pulse crop that is an important source of plant protein.",

    "kidneybeans": "A legume crop commonly used as a food source.",

    "pigeonpeas": "A pulse crop widely used in traditional food.",

    "mothbeans": "A drought-tolerant legume crop.",

    "mungbean": "A short-duration pulse crop.",

    "blackgram": "A pulse crop commonly used in Indian cooking.",

    "lentil": "A protein-rich pulse crop.",

    "pomegranate": "A fruit crop known for its red, juicy seeds.",

    "banana": "A tropical fruit crop.",

    "mango": "A popular tropical fruit crop.",

    "grapes": "A fruit crop used for fresh consumption and processing.",

    "watermelon": "A fruit crop with high water content.",

    "muskmelon": "A sweet fruit crop commonly grown in warm conditions.",

    "apple": "A fruit crop grown in suitable temperate conditions.",

    "orange": "A citrus fruit crop rich in vitamin C.",

    "papaya": "A tropical fruit crop.",

    "coconut": "A tropical plantation crop with multiple uses.",

    "cotton": "A commercial crop used in the textile industry.",

    "jute": "A natural fibre crop.",

    "coffee": "A plantation crop used to produce coffee beans."

}


# ==================================================
# 7. CROP EMOJIS
# ==================================================

crop_emoji = {

    "rice": "🌾",
    "maize": "🌽",
    "chickpea": "🫘",
    "kidneybeans": "🫘",
    "pigeonpeas": "🫘",
    "mothbeans": "🌱",
    "mungbean": "🌱",
    "blackgram": "🌱",
    "lentil": "🫘",
    "pomegranate": "🍎",
    "banana": "🍌",
    "mango": "🥭",
    "grapes": "🍇",
    "watermelon": "🍉",
    "muskmelon": "🍈",
    "apple": "🍎",
    "orange": "🍊",
    "papaya": "🥭",
    "coconut": "🥥",
    "cotton": "🌿",
    "jute": "🌿",
    "coffee": "☕"

}


# ==================================================
# 8. SESSION STATE
# ==================================================

if "history" not in st.session_state:

    st.session_state.history = []


# Set default input values

default_values = {

    "N": 0.0,
    "P": 0.0,
    "K": 0.0,
    "temperature": 0.0,
    "humidity": 0.0,
    "ph": 0.0,
    "rainfall": 0.0

}

for key, value in default_values.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ==================================================
# 9. HEADER
# ==================================================

st.markdown(
    """
    <section class="hero">
        <div class="hero-glow-orb" aria-hidden="true"></div>
        <div class="hero-content">
            <div class="hero-badge">
                <span class="pulse-dot"></span>
                <span class="hero-eyebrow">AI Agronomy · Smart Soil & Climate Match</span>
            </div>
            <h1>Know what to grow.</h1>
            <p>Match your soil and local weather to a crop recommendation,
            with all supported crops ranked for comparison.</p>
        </div>
        <div class="hero-mark-wrap" aria-hidden="true">
            <div class="hero-mark-ring"></div>
            <div class="hero-mark">🌱</div>
        </div>
    </section>
    """,
    unsafe_allow_html=True
)


# ==================================================
# 10. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.markdown(
    '<div class="sidebar-brand">🌱 CropWise</div>',
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Crop Recommendation",
        "Prediction History"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Crop suggestions based on your soil and local weather."
)


# ==================================================
# 11. CROP RECOMMENDATION PAGE
# ==================================================

if page == "Crop Recommendation":

    st.markdown(
        '<div class="section-title">Your farm snapshot</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Use your latest soil test and local weather readings for the best match."
    )


    # ----------------------------------------------
    # SAMPLE VALUES BUTTON
    # ----------------------------------------------

    if st.button("✨ Try Sample Values"):

        st.session_state.N = 90.0
        st.session_state.P = 42.0
        st.session_state.K = 43.0

        st.session_state.temperature = 20.8
        st.session_state.humidity = 82.0

        st.session_state.ph = 6.5
        st.session_state.rainfall = 202.9

        st.rerun()


    # ----------------------------------------------
    # SOIL NUTRIENTS
    # ----------------------------------------------

    with st.container(border=True):
        st.markdown(
            '<div class="section-heading"><span class="step-number">01</span> Soil nutrients</div>',
            unsafe_allow_html=True
        )
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            nitrogen = st.number_input(
                "Nitrogen (N)",
                min_value=0.0,
                key="N"
            )

        with col2:
            phosphorus = st.number_input(
                "Phosphorus (P)",
                min_value=0.0,
                key="P"
            )

        with col3:
            potassium = st.number_input(
                "Potassium (K)",
                min_value=0.0,
                key="K"
            )

        with col4:
            ph = st.number_input(
                "Soil pH",
                min_value=0.0,
                max_value=14.0,
                key="ph"
            )

    with st.container(border=True):
        st.markdown(
            '<div class="section-heading"><span class="step-number">02</span> Weather conditions</div>',
            unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns(3)

        with col1:
            temperature = st.number_input(
                "Temperature (°C)",
                min_value=-50.0,
                max_value=70.0,
                key="temperature"
            )

        with col2:
            humidity = st.number_input(
                "Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                key="humidity"
            )

        with col3:
            rainfall = st.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                key="rainfall"
            )


    # ----------------------------------------------
    # RECOMMENDATION BUTTON
    # ----------------------------------------------

    if st.button(
        "Get my crop recommendation",
        type="primary",
        width="stretch"
    ):

        # Prepare input data

        input_data = pd.DataFrame(
            [[
                nitrogen,
                phosphorus,
                potassium,
                temperature,
                humidity,
                ph,
                rainfall
            ]],

            columns=[
                "N",
                "P",
                "K",
                "temperature",
                "humidity",
                "ph",
                "rainfall"
            ]
        )


        # ------------------------------------------
        # CHECK INPUT VALUES
        # ------------------------------------------

        if hasattr(model, "feature_names_in_"):

            expected_columns = list(model.feature_names_in_)

            if expected_columns != list(input_data.columns):

                st.error(
                    "Input feature names or order do not match "
                    "the trained model."
                )

                st.write("Model expects:", expected_columns)

                st.stop()


        # ------------------------------------------
        # CHECK DATASET RANGES
        # ------------------------------------------

        if dataset is not None:

            out_of_range = []

            for column in input_data.columns:

                minimum = dataset[column].min()

                maximum = dataset[column].max()

                value = input_data[column].iloc[0]

                if value < minimum or value > maximum:

                    out_of_range.append(
                        f"{column}: observed training range "
                        f"{minimum:.2f} to {maximum:.2f}"
                    )


            if out_of_range:

                st.warning(
                    "Some values are outside the dataset's "
                    "observed ranges. Predictions may be less reliable."
                )

                for warning in out_of_range:

                    st.write("•", warning)


        # ------------------------------------------
        # MAKE CROP PREDICTION
        # ------------------------------------------

        try:

            prediction = model.predict(input_data)[0]

            # Get prediction probabilities

            if not hasattr(model, "predict_proba"):

                st.error(
                    "The loaded model does not support prediction probabilities."
                )

                st.stop()


            probabilities = model.predict_proba(input_data)[0]

            confidence = probabilities.max() * 100


            # --------------------------------------
            # ALL CROP PREDICTIONS
            # --------------------------------------

            crop_probabilities = sorted(

                zip(model.classes_, probabilities),

                key=lambda x: x[1],

                reverse=True

            )


            # --------------------------------------
            # CROP INFORMATION
            # --------------------------------------

            description = crop_info.get(
                prediction,
                "Crop information is not available."
            )

            emoji = crop_emoji.get(
                prediction,
                "🌱"
            )


            # --------------------------------------
            # DISPLAY MAIN RESULT
            # --------------------------------------

            st.markdown(
                f"""
                <section class="result-card" aria-label="Crop recommendation">
                    <div class="result-badge-row">
                        <span class="result-eyebrow">✨ Recommended Crop</span>
                    </div>
                    <div class="crop-header-wrap">
                        <span class="crop-emoji-animated">{emoji}</span>
                        <h2 class="crop-name">{prediction.title()}</h2>
                    </div>
                    <p class="result-description">{description}</p>
                </section>
                """,
                unsafe_allow_html=True
            )


            # --------------------------------------
            # SAVE PREDICTION HISTORY
            # --------------------------------------

            st.session_state.history.append({

                "Crop": prediction.title(),

                "Confidence": round(confidence, 2),

                "Nitrogen": nitrogen,

                "Phosphorus": phosphorus,

                "Potassium": potassium,

                "Temperature": temperature,

                "Humidity": humidity,

                "pH": ph,

                "Rainfall": rainfall

            })


            with st.expander(
                f"Compare all {len(crop_probabilities)} crop matches"
            ):
                crop_probability_df = pd.DataFrame(
                    crop_probabilities,
                    columns=["Crop", "Probability"]
                )
                crop_probability_df["Crop"] = (
                    crop_probability_df["Crop"].str.title()
                )
                crop_probability_df["Probability (%)"] = (
                    crop_probability_df.pop("Probability") * 100
                )
                crop_probability_df = crop_probability_df.sort_values(
                    "Probability (%)",
                    ascending=True
                )

                bar_colors = [
                    "#10b981" if crop == prediction.title()
                    else "#c8e7cf"
                    for crop in crop_probability_df["Crop"]
                ]
                fig, ax = plt.subplots(figsize=(11, 9))
                fig.patch.set_alpha(0.0)
                ax.set_facecolor("#ffffff")
                bars = ax.barh(
                    crop_probability_df["Crop"],
                    crop_probability_df["Probability (%)"],
                    color=bar_colors,
                    height=0.72
                )
                maximum_probability = crop_probability_df[
                    "Probability (%)"
                ].max()
                ax.set_xlim(0, max(1, maximum_probability * 1.15))
                for bar, probability, crop in zip(
                    bars,
                    crop_probability_df["Probability (%)"],
                    crop_probability_df["Crop"]
                ):
                    is_best = (crop == prediction.title())
                    ax.annotate(
                        f"{probability:.2f}%",
                        xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                        xytext=(6, 0),
                        textcoords="offset points",
                        va="center",
                        fontsize=8.5,
                        fontweight="bold" if is_best else "normal",
                        color="#047857" if is_best else "#52675a"
                    )

                ax.set_title("Crop Match Probabilities", fontsize=12, fontweight="bold", color="#122b1e", pad=12)
                ax.set_xlabel("Predicted Probability (%)", fontsize=9.5, fontweight="600", color="#3b5647")
                ax.set_axisbelow(True)
                ax.xaxis.grid(color="#e2eae2", linewidth=0.8, linestyle="--")
                ax.tick_params(axis="both", length=0, colors="#3b5647", labelsize=9)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            with st.expander("Review the values used"):
                st.dataframe(
                    input_data,
                    width="stretch",
                    hide_index=True
                )

            st.info(
                "Use actual soil laboratory results for N, P, K "
                "and pH. This recommendation is preliminary and "
                "should be considered along with local agricultural advice."
            )


        except Exception as error:

            st.error("Unable to generate crop prediction.")

            st.code(str(error))


# ==================================================
# 12. PREDICTION HISTORY PAGE
# ==================================================

elif page == "Prediction History":

    st.markdown(
        '<div class="section-title">📜 Prediction History</div>',
        unsafe_allow_html=True
    )


    if len(st.session_state.history) > 0:

        history_df = pd.DataFrame(
            st.session_state.history
        )


        st.metric(
            "Total Predictions",
            len(history_df)
        )


        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True
        )


        if st.button("🗑️ Clear History"):

            st.session_state.history = []

            st.rerun()


    else:

        st.info(
            "No prediction history available. "
            "Make your first crop recommendation!"
        )


# ==================================================
# 13. FOOTER
# ==================================================

st.divider()

st.markdown(
    """
    <div class="footer-note">
        🌱 CropWise · Growing smarter decisions from farm data
    </div>
    """,
    unsafe_allow_html=True
)
