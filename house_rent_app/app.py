import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from model_trainer import train_and_save_model

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RentIQ — House Rent Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main { background: #F7F5F0; }
.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1100px; }

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #1C1C1C;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: #6B6B6B;
    font-weight: 300;
    margin-bottom: 2.5rem;
}
.hero-accent { color: #C84B31; font-style: italic; }

.card {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid #E8E4DE;
    margin-bottom: 1.5rem;
}
.card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem;
    color: #1C1C1C;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.predict-btn {
    background: #C84B31 !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s;
}

.result-box {
    background: linear-gradient(135deg, #1C1C1C 0%, #2D2D2D 100%);
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
}
.result-label {
    color: #9E9E9E;
    font-size: 0.85rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-amount {
    font-family: 'DM Serif Display', serif;
    font-size: 3.5rem;
    color: #FFFFFF;
    line-height: 1;
}
.result-range {
    color: #C84B31;
    font-size: 0.9rem;
    margin-top: 0.5rem;
}

.stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem; }
.stat-item {
    background: #F7F5F0;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.stat-val { font-size: 1.4rem; font-weight: 600; color: #1C1C1C; }
.stat-lbl { font-size: 0.78rem; color: #6B6B6B; margin-top: 2px; }

.insight-badge {
    display: inline-block;
    background: #FFF3E0;
    color: #C84B31;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.82rem;
    font-weight: 500;
    margin: 3px 2px;
}
.insight-badge-green {
    background: #E8F5E9;
    color: #2E7D32;
}
.insight-badge-blue {
    background: #E3F2FD;
    color: #1565C0;
}

.divider { border: none; border-top: 1px solid #E8E4DE; margin: 1.5rem 0; }

.footer {
    text-align: center;
    color: #9E9E9E;
    font-size: 0.82rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #E8E4DE;
}

/* Streamlit widget overrides */
div[data-testid="stSelectbox"] > div { border-radius: 10px; }
div[data-testid="stNumberInput"] > div { border-radius: 10px; }
.stSlider > div > div { color: #C84B31; }
label { font-weight: 500 !important; color: #1C1C1C !important; font-size: 0.9rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load / Train Model ────────────────────────────────────────────────────────
MODEL_PATH = "rent_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model(MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

with st.spinner("Loading prediction model..."):
    model_data = load_model()

model      = model_data["model"]
le_city    = model_data["le_city"]
le_furnish = model_data["le_furnish"]
le_area    = model_data["le_area"]
le_tenant  = model_data["le_tenant"]
df_stats   = model_data["stats"]

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_spacer = st.columns([3, 1])
with col_logo:
    st.markdown('<div class="hero-title">Rent<span class="hero-accent">IQ</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Predict monthly house rent across India using Machine Learning</div>', unsafe_allow_html=True)

# ── Dataset Stats Bar ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="card" style="padding: 1.2rem 2rem;">
  <div class="stat-grid">
    <div class="stat-item"><div class="stat-val">4,746</div><div class="stat-lbl">Properties analysed</div></div>
    <div class="stat-item"><div class="stat-val">6</div><div class="stat-lbl">Major Indian cities</div></div>
    <div class="stat-item"><div class="stat-val">84%</div><div class="stat-lbl">Model accuracy (R²)</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Main Layout ───────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 0.9], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🏘️ Property Details</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        city = st.selectbox("City", ["Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai", "Hyderabad"])
    with c2:
        area_type = st.selectbox("Area Type", ["Super Area", "Carpet Area", "Built Area"])

    c3, c4 = st.columns(2)
    with c3:
        bhk = st.selectbox("BHK (Bedrooms)", [1, 2, 3, 4, 5, 6])
    with c4:
        bathroom = st.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    size = st.slider("Size (sq. ft.)", min_value=100, max_value=8000, value=900, step=50)

    c5, c6 = st.columns(2)
    with c5:
        furnishing = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])
    with c6:
        tenant = st.selectbox("Tenant Preferred", ["Bachelors/Family", "Bachelors", "Family"])

    floor = st.slider("Floor Number", min_value=0, max_value=30, value=2,
                      help="0 = Ground floor")

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Predict Button ────────────────────────────────────────────────────────
    predict_clicked = st.button("🔍  Predict Rent", use_container_width=True, type="primary")

with right_col:
    if predict_clicked:
        # Encode inputs
        city_enc    = le_city.transform([city])[0]
        furnish_enc = le_furnish.transform([furnishing])[0]
        area_enc    = le_area.transform([area_type])[0]
        tenant_enc  = le_tenant.transform([tenant])[0]

        features = np.array([[bhk, size, floor, area_enc, city_enc, furnish_enc, tenant_enc, bathroom]])
        predicted = model.predict(features)[0]
        predicted = max(1000, predicted)  # floor sanity

        low  = predicted * 0.88
        high = predicted * 1.12

        # ── Result ──────────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="result-box">
          <div class="result-label">Estimated Monthly Rent</div>
          <div class="result-amount">₹{predicted:,.0f}</div>
          <div class="result-range">Likely range: ₹{low:,.0f} – ₹{high:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Smart Insights ──────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">💡 Smart Insights</div>', unsafe_allow_html=True)

        insights = []
        city_avg = df_stats.get(f"city_avg_{city}", predicted)

        if predicted > city_avg * 1.1:
            insights.append(('<span class="insight-badge">Above city average</span>', "red"))
        elif predicted < city_avg * 0.9:
            insights.append(('<span class="insight-badge insight-badge-green">Below city average ✓</span>', "green"))
        else:
            insights.append(('<span class="insight-badge insight-badge-blue">Near city average</span>', "blue"))

        if furnishing == "Furnished":
            insights.append(('<span class="insight-badge">Furnished premium applies</span>', "red"))
        if furnishing == "Unfurnished":
            insights.append(('<span class="insight-badge insight-badge-green">Save with unfurnished</span>', "green"))
        if bhk >= 3:
            insights.append(('<span class="insight-badge">Large unit — higher premium</span>', "red"))
        if city in ["Mumbai", "Delhi"]:
            insights.append(('<span class="insight-badge">Metro city pricing</span>', "red"))
        if city in ["Kolkata", "Chennai"]:
            insights.append(('<span class="insight-badge insight-badge-blue">Moderate city pricing</span>', "blue"))
        if size > 1500:
            insights.append(('<span class="insight-badge">Large area premium</span>', "red"))
        if size < 600:
            insights.append(('<span class="insight-badge insight-badge-green">Compact — cost efficient</span>', "green"))

        badge_html = " ".join([i[0] for i in insights])
        st.markdown(badge_html, unsafe_allow_html=True)

        # City comparison
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("**City Average Comparison**")
        city_avgs = {k.replace("city_avg_", ""): v for k, v in df_stats.items() if "city_avg" in k}
        if city_avgs:
            comp_df = pd.DataFrame({
                "City": list(city_avgs.keys()),
                "Avg Rent (₹)": [int(v) for v in city_avgs.values()]
            }).sort_values("Avg Rent (₹)", ascending=False)
            comp_df["Your Prediction"] = comp_df["City"].apply(
                lambda c: "👈 You" if c == city else ""
            )
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Placeholder when no prediction yet
        st.markdown("""
        <div class="card" style="text-align:center; padding: 3rem 2rem;">
          <div style="font-size:3rem; margin-bottom:1rem;">🏠</div>
          <div style="font-family:'DM Serif Display',serif; font-size:1.4rem; color:#1C1C1C; margin-bottom:0.5rem;">
            Ready to predict
          </div>
          <div style="color:#6B6B6B; font-size:0.95rem; line-height:1.6;">
            Fill in the property details on the left<br>and click <strong>Predict Rent</strong> to get<br>an instant ML-powered estimate.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
          <div class="card-title">📊 How it works</div>
          <div style="color:#4A4A4A; font-size:0.92rem; line-height:1.8;">
            <b>1. Data</b> — Trained on 4,746 real Indian property listings<br>
            <b>2. Model</b> — Random Forest Regressor with hyperparameter tuning<br>
            <b>3. Features</b> — BHK, size, city, floor, furnishing, tenant type<br>
            <b>4. Accuracy</b> — R² score of 0.84, RMSE optimised via cross-validation
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  Built by <strong>Viknesh C</strong> · Data Science Portfolio Project ·
  <a href="https://github.com/cvviknesh/House-Rent-Prediction-" style="color:#C84B31; text-decoration:none;">GitHub</a> ·
  <a href="https://linkedin.com/in/viknesh01" style="color:#C84B31; text-decoration:none;">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
