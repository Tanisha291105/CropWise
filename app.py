
import streamlit as st
import pandas as pd
import joblib
import os


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
# 3. CUSTOM CSS
# ==================================================

st.markdown("""
<style>

:root {
    --forest: #173d2d;
    --leaf: #2f7048;
    --mint: #e8f2e8;
    --paper: #f5f7f2;
    --ink: #20342a;
    --muted: #64766a;
}

.stApp,
[data-testid="stAppViewContainer"] {
    background: var(--paper);
    color: var(--ink);
}

[data-testid="stHeader"] {
    background: transparent;
}

.main .block-container {
    max-width: 1120px;
    padding: 2rem 2.25rem 3rem;
}

[data-testid="stSidebar"] {
    background: var(--forest);
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #f3f7f1;
}

[data-testid="stSidebar"] [data-testid="stRadio"] > label {
    color: #b6cabc;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.sidebar-brand {
    color: #f4f8f2;
    font-size: 1.45rem;
    font-weight: 750;
    letter-spacing: -0.04em;
    padding: 0.3rem 0 1.25rem;
}

.hero {
    align-items: center;
    background: linear-gradient(120deg, #173d2d 0%, #286343 100%);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 24px;
    color: #f5f8f2;
    display: flex;
    justify-content: space-between;
    margin-bottom: 1.6rem;
    min-height: 190px;
    overflow: hidden;
    padding: 2rem 2.5rem;
}

.hero-eyebrow,
.result-eyebrow {
    color: #b8d5b9;
    font-size: 0.75rem;
    font-weight: 750;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.hero h1 {
    color: #ffffff;
    font-size: clamp(2rem, 4vw, 3rem);
    letter-spacing: -0.055em;
    line-height: 1.08;
    margin: 0.55rem 0;
}

.hero p {
    color: #d7e5d9;
    font-size: 1rem;
    margin: 0;
    max-width: 570px;
}

.hero-mark {
    align-items: center;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.16);
    border-radius: 22px;
    display: flex;
    font-size: 3.2rem;
    height: 96px;
    justify-content: center;
    margin-left: 1.5rem;
    min-width: 96px;
}

.section-title {
    color: var(--forest);
    font-size: 1.45rem;
    font-weight: 750;
    letter-spacing: -0.035em;
}

.section-heading {
    align-items: center;
    color: var(--forest);
    display: flex;
    font-size: 1.08rem;
    font-weight: 700;
    gap: 0.65rem;
    margin-bottom: 1.1rem;
}

.step-number {
    align-items: center;
    background: var(--mint);
    border-radius: 9px;
    color: var(--leaf);
    display: inline-flex;
    font-size: 0.78rem;
    height: 29px;
    justify-content: center;
    width: 29px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border: 1px solid #e2e9df;
    border-radius: 18px;
}

[data-testid="stNumberInput"] input {
    background: #fbfcfa;
    border-radius: 10px;
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink);
}

[data-testid="stNumberInput"] label {
    color: #42584a;
    font-weight: 600;
}

div.stButton > button {
    border-radius: 11px;
    font-weight: 650;
    min-height: 2.8rem;
    transition: transform 120ms ease, box-shadow 120ms ease;
}

div.stButton > button[kind="primary"] {
    background: var(--leaf);
    border: 1px solid var(--leaf);
    box-shadow: 0 5px 14px rgba(47, 112, 72, 0.16);
    color: #ffffff;
    width: 100%;
}

div.stButton > button[kind="primary"]:hover {
    background: var(--forest);
    border-color: var(--forest);
    color: #ffffff;
    transform: translateY(-1px);
}

div.stButton > button:not([kind="primary"]) {
    background: #ffffff;
    border: 1px solid #dce6da;
    color: var(--forest);
}

.result-card {
    background: linear-gradient(135deg, #e9f4e8 0%, #f7faf4 100%);
    border: 1px solid #d5e6d3;
    border-left: 5px solid #4d9560;
    border-radius: 18px;
    margin: 1rem 0 1.3rem;
    padding: 1.5rem 1.7rem;
}

.result-eyebrow {
    color: #477754;
}

.crop-name {
    color: var(--forest);
    font-size: clamp(2rem, 4vw, 2.6rem);
    font-weight: 780;
    letter-spacing: -0.05em;
    margin: 0.25rem 0;
}

.result-description {
    color: #52675a;
    font-size: 1rem;
    margin: 0.45rem 0 1.15rem;
}

.confidence-row {
    align-items: center;
    border-top: 1px solid #d5e6d3;
    color: #52675a;
    display: flex;
    justify-content: space-between;
    padding-top: 0.9rem;
}

.confidence-value {
    color: var(--forest);
    font-size: 1.1rem;
    font-weight: 750;
}

.footer-note {
    color: var(--muted);
    font-size: 0.85rem;
    padding: 0.4rem 0 1rem;
    text-align: center;
}

@media (max-width: 640px) {
    .main .block-container {
        padding: 1.1rem 1rem 2rem;
    }

    .hero {
        border-radius: 18px;
        min-height: 160px;
        padding: 1.5rem;
    }

    .hero-mark {
        font-size: 2rem;
        height: 64px;
        min-width: 64px;
    }
}

</style>
""", unsafe_allow_html=True)


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
        <div>
            <div class="hero-eyebrow">CropWise · Smart farming</div>
            <h1>Know what to grow.</h1>
            <p>Match your soil and local weather to a crop recommendation,
            with all supported crops ranked for comparison.</p>
        </div>
        <div class="hero-mark" aria-hidden="true">🌱</div>
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
                    <div class="result-eyebrow">Your best crop match</div>
                    <div class="crop-name">{emoji} {prediction.title()}</div>
                    <p class="result-description">{description}</p>
                    <div class="confidence-row">
                        <span>Model confidence</span>
                        <span class="confidence-value">{confidence:.2f}%</span>
                    </div>
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


            st.caption(
                "Confidence is the model's estimate, not a guarantee of crop success."
            )

            crop_probability_df = pd.DataFrame(
                crop_probabilities,
                columns=["Crop", "Probability"]
            )
            crop_probability_df["Crop"] = (
                crop_probability_df["Crop"].str.title()
            )
            crop_probability_df["Probability"] = (
                crop_probability_df["Probability"] * 100
            ).round(2)
            crop_probability_df = crop_probability_df.rename(
                columns={"Probability": "Probability (%)"}
            )

            with st.expander(
                f"Compare all {len(crop_probabilities)} crop matches"
            ):
                st.dataframe(
                    crop_probability_df,
                    width="stretch",
                    hide_index=True,
                    height=420
                )

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
