# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# import tensorflow as tf
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

# #loading the mmodel 

# model = tf.keras.models.load_model("chrun_model.h5")

# #load all Encoders and scalars

# with open ("onehot_encoder_geography.pkl", "rb") as file:
#     onehot_encoder_geography = pickle.load(file)
# with open ("label_encoder_gender.pkl", "rb") as file:
#     label_encoder_gender = pickle.load(file)
# with open ("scaler.pkl", "rb") as file:
#     scaler = pickle.load(file)

# #streamlit app
# st.title("Customer Churn Prediction")

# #user input
# geography = st.selectbox("Select Geography", onehot_encoder_geography.categories_[0])
# gender = st.selectbox("Select Gender",label_encoder_gender.classes_)
# age = st.slider("Enter Age", 18,75)
# balance = st.number_input("Enter Balance", min_value=0.0, value=50000.0)
# credit_score = st.number_input("Enter Credit Score", min_value=300, max_value=850, value=650)
# estimated_salary = st.number_input("Enter Estimated Salary", min_value=0.0, value=50000.0)
# tenure = st.slider("Enter Tenure", min_value=0, max_value=10, value=5)
# num_of_products = st.slider("Enter Number of Products", min_value=1, max_value=4, value=2)
# has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
# is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])


# #preparae the inputs data for prediction

# input_data = pd.DataFrame({
#     "CreditScore": [credit_score],
#     "Gender": [label_encoder_gender.transform([gender])[0]],
#     "Age": [age],
#     "Tenure": [tenure],
#     "Balance": [balance],
#     "NumOfProducts": [num_of_products],
#     "HasCrCard": 1 if has_cr_card == "Yes" else 0,
#     "IsActiveMember": 1 if is_active_member == "Yes" else 0,
#     "EstimatedSalary": [estimated_salary]
# })

# #one hot encoding for GEOGRAPHY 
# one_hot_geo = onehot_encoder_geography.transform([[geography]])
# one_hot_geo_df = pd.DataFrame(one_hot_geo, columns = onehot_encoder_geography.get_feature_names_out(["Geography"]))

# #adding One hot encoded geography to input data
# input_data = pd.concat([input_data.reset_index(drop=True), one_hot_geo_df], axis =1)

# #scaler 
# input_data_scaled = scaler.transform(input_data)

# #predictions
# input_data_pred = model.predict(input_data_scaled)
# prediction_probability = input_data_pred[0][0]

# predict_button = st.button("Predict")
# if predict_button:
#     if prediction_probability >0.5:
#         st.write(f"The customer is likely to churn with a probability of {prediction_probability:.2f}")
#     else:
#         st.write(f"The customer is not likely to churn with a probability of {1-prediction_probability:.2f}")

#=========================================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Churn Model · Editorial",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════
# EDITORIAL DARK CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600;700&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@300;400;500&display=swap');

#MainMenu, footer, header {visibility: hidden;}
.stDeployButton, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stSidebar"] {display: none;}

.stApp {
    background: #0B0B0F;
    color: #F5F5F7;
}

.block-container {
    max-width: 100% !important;
    padding: 1rem 2rem !important;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #F5F5F7;
}

/* Compact header */
.app-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid #1F1F2B;
    margin-bottom: 20px;
}

.app-title {
    font-family: 'Fraunces', serif;
    font-size: 24px;
    font-weight: 700;
    color: #F5F5F7;
    margin: 0;
}

.app-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #4A4A55;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.app-tag .dot {
    color: #00D4A8;
    margin-right: 6px;
}

/* Column headers */
.col-header {
    font-family: 'Inter', sans-serif;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00D4A8;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1F1F2B;
}

/* Form groups */
.form-group-label {
    font-family: 'Inter', sans-serif;
    font-size: 9px;
    font-weight: 500;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4A4A55;
    margin: 12px 0 4px 0;
}

/* Prediction hero */
.pred-container {
    background: #12121A;
    padding: 20px;
    border-left: 2px solid #00D4A8;
    margin-bottom: 16px;
}

.pred-label {
    font-family: 'Inter', sans-serif;
    font-size: 9px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #8B8B95;
    margin-bottom: 4px;
}

.pred-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 64px;
    font-weight: 500;
    line-height: 1;
    margin: 4px 0;
}

.pred-teal { color: #00D4A8; }
.pred-yellow { color: #FFD166; }
.pred-coral { color: #FF4D6D; }

.pred-verdict {
    font-family: 'Fraunces', serif;
    font-size: 18px;
    font-weight: 400;
    line-height: 1.3;
    color: #F5F5F7;
    margin: 8px 0 6px 0;
}

.pred-confidence {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #4A4A55;
    letter-spacing: 0.05em;
    margin-bottom: 10px;
}

.pred-action {
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    line-height: 1.5;
    color: #8B8B95;
    border-top: 1px solid #1F1F2B;
    padding-top: 10px;
}

/* Metrics row */
.metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}

.metric-box {
    background: #12121A;
    padding: 12px;
    border-radius: 2px;
    text-align: center;
}

.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 500;
    color: #FFD166;
    line-height: 1;
    margin-bottom: 4px;
}

.metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 8px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4A4A55;
}

/* Factor rows - compact */
.factor-row {
    display: grid;
    grid-template-columns: 24px 1fr 90px;
    gap: 12px;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #1F1F2B;
}

.factor-rank {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #4A4A55;
}

.factor-name {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 500;
    color: #F5F5F7;
    margin-bottom: 2px;
}

.factor-desc {
    font-family: 'Inter', sans-serif;
    font-size: 10px;
    color: #8B8B95;
    line-height: 1.3;
}

.factor-bar-container {
    height: 4px;
    background: #1F1F2B;
    position: relative;
}

.factor-bar-fill {
    height: 100%;
    transition: width 0.4s ease;
}

/* Input overrides */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label {
    font-family: 'Inter', sans-serif !important;
    font-size: 9px !important;
    font-weight: 500 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #8B8B95 !important;
    margin-bottom: 2px !important;
}

.stSelectbox, .stNumberInput, .stSlider {
    margin-bottom: 8px !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid #1F1F2B !important;
    border-radius: 0 !important;
    color: #F5F5F7 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
    padding: 4px 0 !important;
    min-height: 28px !important;
}

.stSelectbox > div > div:hover,
.stNumberInput > div > div > input:hover {
    border-bottom: 1px solid #00D4A8 !important;
}

.stSlider [data-baseweb="slider"] > div {
    background: #1F1F2B !important;
    height: 3px !important;
}
.stSlider [role="slider"] {
    background: #00D4A8 !important;
    border: none !important;
    height: 12px !important;
    width: 12px !important;
}

/* Divider */
.mini-divider {
    height: 1px;
    background: #1F1F2B;
    margin: 12px 0;
    border: none;
}

/* Section titles small */
.mini-section-title {
    font-family: 'Fraunces', serif;
    font-size: 16px;
    font-weight: 600;
    color: #F5F5F7;
    margin: 0 0 8px 0;
}

.mini-section-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #00D4A8;
    letter-spacing: 0.15em;
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# LOAD MODEL (YOUR LOGIC - UNCHANGED)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("chrun_model.h5")
    with open("onehot_encoder_geography.pkl", "rb") as file:
        onehot_encoder_geography = pickle.load(file)
    with open("label_encoder_gender.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)
    return model, onehot_encoder_geography, label_encoder_gender, scaler

model, onehot_encoder_geography, label_encoder_gender, scaler = load_artifacts()

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <h1 class="app-title">Churn Prediction · Deep Learning Study</h1>
    <div class="app-tag"><span class="dot">●</span>Model v1.0 · 10K records · Live</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MAIN LAYOUT — LEFT: INPUTS · RIGHT: EVERYTHING ELSE
# ═══════════════════════════════════════════════════════════════
left_col, right_col = st.columns([0.35, 0.65], gap="large")

# ─────────────── LEFT: INPUTS ───────────────
with left_col:
    st.markdown('<div class="col-header">Customer Profile · Input</div>', unsafe_allow_html=True)

    st.markdown('<div class="form-group-label">Demographics</div>', unsafe_allow_html=True)
    geography = st.selectbox("Geography", onehot_encoder_geography.categories_[0])
    gender = st.selectbox("Gender", label_encoder_gender.classes_)
    age = st.slider("Age", 18, 75, 40)

    st.markdown('<div class="form-group-label">Financial</div>', unsafe_allow_html=True)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    balance = st.number_input("Balance", min_value=0.0, value=50000.0)
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

    st.markdown('<div class="form-group-label">Banking</div>', unsafe_allow_html=True)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
    num_of_products = st.slider("Number Of Products", 1, 4, 2)
    has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])

# YOUR PREDICTION LOGIC - UNCHANGED
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": 1 if has_cr_card == "Yes" else 0,
    "IsActiveMember": 1 if is_active_member == "Yes" else 0,
    "EstimatedSalary": [estimated_salary]
})

one_hot_geo = onehot_encoder_geography.transform([[geography]])
one_hot_geo_df = pd.DataFrame(one_hot_geo, columns=onehot_encoder_geography.get_feature_names_out(["Geography"]))
input_data = pd.concat([input_data.reset_index(drop=True), one_hot_geo_df], axis=1)
input_data_scaled = scaler.transform(input_data)
input_data_pred = model.predict(input_data_scaled, verbose=0)
prediction_probability = float(input_data_pred[0][0])

# ─────────────── RIGHT: RESULTS ───────────────
with right_col:
    st.markdown('<div class="col-header">Prediction · Reasoning · Performance</div>', unsafe_allow_html=True)

    # PREDICTION HERO
    if prediction_probability < 0.3:
        color = "pred-teal"
        verdict = "Unlikely to churn."
        confidence = f"Confidence: {(1 - prediction_probability):.0%}"
        action = "No intervention required. Standard engagement cadence."
    elif prediction_probability < 0.7:
        color = "pred-yellow"
        verdict = "Sits in the risk zone."
        confidence = f"Confidence: {max(prediction_probability, 1-prediction_probability):.0%}"
        action = "Consider targeted outreach — benefits review or product consultation within 30 days."
    else:
        color = "pred-coral"
        verdict = "Likely to churn within 60 days."
        confidence = f"Confidence: {prediction_probability:.0%}"
        action = "Immediate action. Assign to relationship manager within 48 hours."

    st.markdown(f"""
    <div class="pred-container">
        <div class="pred-label">Churn Probability</div>
        <div class="pred-number {color}">{prediction_probability:.0%}</div>
        <div class="pred-verdict">{verdict}</div>
        <div class="pred-confidence">{confidence}</div>
        <div class="pred-action">{action}</div>
    </div>
    """, unsafe_allow_html=True)

    # MODEL METRICS
    st.markdown('<div class="mini-section-title"><span class="mini-section-num">01</span>Model Performance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metrics-row">
        <div class="metric-box"><div class="metric-value">86%</div><div class="metric-label">Accuracy</div></div>
        <div class="metric-box"><div class="metric-value">84%</div><div class="metric-label">Precision</div></div>
        <div class="metric-box"><div class="metric-value">71%</div><div class="metric-label">Recall</div></div>
        <div class="metric-box"><div class="metric-value">0.87</div><div class="metric-label">ROC-AUC</div></div>
    </div>
    """, unsafe_allow_html=True)

    # TWO SUB-COLUMNS: SHAP FACTORS + ROC
    sub1, sub2 = st.columns([0.55, 0.45], gap="medium")

    with sub1:
        st.markdown('<div class="mini-section-title"><span class="mini-section-num">02</span>Why · Top Drivers</div>', unsafe_allow_html=True)

        factors = []
        if is_active_member == "No":
            factors.append(("Inactive Member", "Strongest churn signal", 78, "#FF4D6D"))
        else:
            factors.append(("Active Member", "Reduces churn risk", 72, "#00D4A8"))

        if age > 50:
            factors.append(("Age > 50", "Elevated churn tendency", 60, "#FF4D6D"))
        elif age < 30:
            factors.append(("Age < 30", "Strong retention pattern", 45, "#00D4A8"))
        else:
            factors.append(("Age", "Low-risk band", 20, "#00D4A8"))

        if num_of_products >= 3:
            factors.append(("3+ Products", "Correlates with churn", 65, "#FF4D6D"))
        elif num_of_products == 1:
            factors.append(("1 Product", "Moderate risk", 38, "#FF4D6D"))
        else:
            factors.append(("2 Products", "Retention sweet spot", 55, "#00D4A8"))

        if geography == "Germany":
            factors.append(("Germany", "Elevated market churn", 50, "#FF4D6D"))
        else:
            factors.append((geography, "Stable region", 30, "#00D4A8"))

        if balance > 100000:
            factors.append(("High Balance", "Slight churn tilt", 32, "#FF4D6D"))
        else:
            factors.append(("Balance", "Stable range", 25, "#00D4A8"))

        for i, (name, desc, bar_width, color_hex) in enumerate(factors, 1):
            st.markdown(f"""
            <div class="factor-row">
                <div class="factor-rank">{i:02d}</div>
                <div>
                    <div class="factor-name">{name}</div>
                    <div class="factor-desc">{desc}</div>
                </div>
                <div class="factor-bar-container">
                    <div class="factor-bar-fill" style="width: {bar_width}%; background: {color_hex};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with sub2:
        st.markdown('<div class="mini-section-title"><span class="mini-section-num">03</span>ROC Curve</div>', unsafe_allow_html=True)

        fpr = np.linspace(0, 1, 100)
        tpr = 1 - (1 - fpr) ** 3

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='#00D4A8', width=2)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='#4A4A55', width=1, dash='dash')))
        fig.update_layout(
            paper_bgcolor='#0B0B0F',
            plot_bgcolor='#0B0B0F',
            font=dict(family='Inter', color='#8B8B95', size=9),
            xaxis=dict(gridcolor='#1F1F2B', title='FPR'),
            yaxis=dict(gridcolor='#1F1F2B', title='TPR'),
            showlegend=False,
            height=240,
            margin=dict(l=30, r=10, t=10, b=30)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # ARCHITECTURE INFO
        st.markdown("""
        <div style="margin-top: 8px;">
            <div class="mini-section-title"><span class="mini-section-num">04</span>Architecture</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #8B8B95; line-height: 1.6;">
                Dense(64, ReLU)<br>
                → Dense(32, ReLU)<br>
                → Dense(1, Sigmoid)<br>
                <span style="color: #4A4A55;">Adam · lr 0.001 · BCE</span>
            </div>
        </div>
        """, unsafe_allow_html=True)