import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Page Configurations ---
st.set_page_config(
    page_title="PrimeTrust | Institutional Loan Decision Engine",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling & CSS Animations ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400..700;1,400..700&family=Inter:wght@300;400;500;600;700&display=swap');

/* Main font styling */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Inter', sans-serif !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Lora', serif !important;
}

/* Institutional Red-White Theme Layout */
.title-container {
    background: #FFFFFF;
    border-top: 5px solid #B91C1C;
    border-bottom: 1px solid #E2E8F0;
    border-left: 1px solid #E2E8F0;
    border-right: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 35px 20px;
    margin-bottom: 30px;
    text-align: center;
    box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.05);
}

.title-header {
    color: #B91C1C;
    font-weight: 700;
    font-size: 2.8rem;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
}

.subtitle-header {
    color: #475569;
    font-size: 0.95rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.15em;
}

/* Glassmorphism containers - styled as classic white banking cards */
.glass-container {
    background: #FFFFFF;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px -2px rgba(0, 0, 0, 0.05);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-container:hover {
    border-color: #B91C1C;
    box-shadow: 0 8px 20px -6px rgba(185, 28, 28, 0.08);
}

.section-title {
    color: #B91C1C;
    font-weight: 600;
    border-bottom: 1px solid #FEE2E2;
    padding-bottom: 8px;
    margin-top: 0;
    margin-bottom: 20px;
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    justify-content: center;
    border-bottom: 2px solid #E2E8F0;
}

.stTabs [data-baseweb="tab"] {
    background-color: #F8FAFC;
    border-radius: 6px 6px 0px 0px;
    padding: 12px 28px;
    color: #64748B;
    border: 1px solid #E2E8F0;
    border-bottom: none;
    transition: all 0.25s ease;
    font-weight: 500;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #B91C1C;
    background-color: #FFF5F5;
}

.stTabs [aria-selected="true"] {
    background-color: #FFFFFF !important;
    color: #B91C1C !important;
    border-top: 3px solid #B91C1C !important;
    border-left: 1px solid #E2E8F0 !important;
    border-right: 1px solid #E2E8F0 !important;
    font-weight: 600;
    box-shadow: none;
}

/* Custom Metrics Design */
.metrics-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
}

.custom-metric {
    flex: 1;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-top: 3px solid #B91C1C;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    transition: all 0.3s ease;
}

.custom-metric:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
}

.metric-val {
    font-family: 'Lora', serif !important;
    font-size: 2.3rem;
    font-weight: 700;
    color: #0F172A;
    margin: 5px 0;
}

.metric-lbl {
    font-size: 0.85rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}

/* Prediction Result Cards */
.result-card {
    border-radius: 8px;
    padding: 30px;
    text-align: center;
    margin-top: 25px;
    border: 1px solid;
    animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05);
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

.approved-card {
    background: #F0FDF4;
    border-color: #BBF7D0;
    border-left: 6px solid #10B981;
}

.rejected-card {
    background: #FEF2F2;
    border-color: #FEE2E2;
    border-left: 6px solid #EF4444;
}

/* Success SVG Checkmark Animation */
.checkmark-wrapper {
    width: 90px;
    height: 90px;
    margin: 0 auto 20px auto;
}

.checkmark {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    display: block;
    stroke-width: 3;
    stroke: #10B981;
    stroke-miterlimit: 10;
    box-shadow: inset 0px 0px 0px rgba(16, 185, 129, 0.1);
    animation: fill-green .4s ease-in-out .4s forwards, scale-up .3s ease-in-out .9s both;
}

.checkmark__circle {
    stroke-dasharray: 166;
    stroke-dashoffset: 166;
    stroke-width: 3;
    stroke-miterlimit: 10;
    stroke: #10B981;
    fill: none;
    animation: stroke 0.6s cubic-bezier(0.65, 0, 0.45, 1) forwards;
}

.checkmark__check {
    transform-origin: 50% 50%;
    stroke-dasharray: 48;
    stroke-dashoffset: 48;
    animation: stroke 0.3s cubic-bezier(0.65, 0, 0.45, 1) 0.6s forwards;
}

/* Error/Cross SVG Animation */
.crossmark-wrapper {
    width: 90px;
    height: 90px;
    margin: 0 auto 20px auto;
}

.crossmark {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    display: block;
    stroke-width: 3;
    stroke: #EF4444;
    stroke-miterlimit: 10;
    box-shadow: inset 0px 0px 0px rgba(239, 68, 68, 0.1);
    animation: fill-red .4s ease-in-out .4s forwards, scale-up .3s ease-in-out .9s both;
}

.crossmark__circle {
    stroke-dasharray: 166;
    stroke-dashoffset: 166;
    stroke-width: 3;
    stroke-miterlimit: 10;
    stroke: #EF4444;
    fill: none;
    animation: stroke 0.6s cubic-bezier(0.65, 0, 0.45, 1) forwards;
}

.crossmark__path {
    transform-origin: 50% 50%;
    stroke-dasharray: 48;
    stroke-dashoffset: 48;
    animation: stroke 0.3s cubic-bezier(0.65, 0, 0.45, 1) 0.6s forwards;
}

@keyframes stroke {
    100% { stroke-dashoffset: 0; }
}
@keyframes scale-up {
    0%, 100% { transform: none; }
    50% { transform: scale(1.08); }
}
@keyframes fill-green {
    100% { box-shadow: inset 0px 0px 0px 45px rgba(16, 185, 129, 0.1); }
}
@keyframes fill-red {
    100% { box-shadow: inset 0px 0px 0px 45px rgba(239, 68, 68, 0.1); }
}

/* Classic Solid Red Button */
.stButton>button {
    background: #B91C1C !important;
    color: white !important;
    border: 1px solid #991B1B !important;
    padding: 12px 30px !important;
    border-radius: 6px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.03em;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 10px rgba(185, 28, 28, 0.15) !important;
    width: 100%;
}

.stButton>button:hover {
    background: #991B1B !important;
    box-shadow: 0 6px 15px rgba(153, 27, 27, 0.3) !important;
    transform: translateY(-1px) !important;
}

.stButton>button:active {
    transform: translateY(1px) !important;
}

/* DTI alerts wrapper */
.dti-wrapper {
    background: #F8FAFC;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    padding: 16px;
    text-align: center;
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)


# --- Helper Function: Generate Dataset if missing ---
def verify_dataset():
    dataset_path = "loan_dataset.csv"
    if not os.path.exists(dataset_path):
        np.random.seed(42)
        n = 1500
        data = pd.DataFrame({
            'ApplicantID': range(1001, 1001+n),
            'Gender': np.random.choice(['Male', 'Female'], n),
            'Age': np.random.randint(21, 60, n),
            'MonthlyIncome': np.random.randint(15000, 150000, n),
            'LoanAmount': np.random.randint(50000, 1500000, n),
            'CreditScore': np.random.randint(300, 900, n),
            'EmploymentStatus': np.random.choice(
                ['Employed', 'Self-Employed', 'Unemployed'],
                n,
                p=[0.6, 0.25, 0.15]
            ),
            'ExistingLoans': np.random.randint(0, 5, n),
            'LoanTerm': np.random.choice([12, 24, 36, 48, 60], n),
            'PropertyArea': np.random.choice(['Urban', 'Semi-Urban', 'Rural'], n)
        })

        # Calculate logical approval rules
        approval = []
        for i in range(n):
            score = 0
            if data.loc[i, 'MonthlyIncome'] > 50000:
                score += 2
            if data.loc[i, 'CreditScore'] > 650:
                score += 3
            if data.loc[i, 'EmploymentStatus'] != 'Unemployed':
                score += 2
            if data.loc[i, 'ExistingLoans'] <= 2:
                score += 1
            if data.loc[i, 'LoanAmount'] < data.loc[i, 'MonthlyIncome'] * 25:
                score += 2
            approval.append(1 if score >= 6 else 0)

        data['LoanApproved'] = approval

        for col in ['Gender', 'CreditScore']:
            data.loc[data.sample(frac=0.02).index, col] = np.nan

        data.to_csv(dataset_path, index=False)


# --- Cached ML Model Training ---
@st.cache_resource(show_spinner="Training Loan Approval Prediction model...")
def load_and_train_model():
    verify_dataset()
    data = pd.read_csv("loan_dataset.csv")

    # Data Cleaning (as in original script)
    data.drop_duplicates(inplace=True)
    data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
    data['CreditScore'].fillna(data['CreditScore'].median(), inplace=True)

    df_clean = data.copy()

    # Manual deterministic mapping matching original maps
    gender_map = {'Male': 1, 'Female': 0}
    employment_map = {'Employed': 0, 'Self-Employed': 1, 'Unemployed': 2}
    property_map = {'Rural': 0, 'Semi-Urban': 1, 'Urban': 2}

    # Encode categorical variables
    data['Gender'] = data['Gender'].map(gender_map)
    data['EmploymentStatus'] = data['EmploymentStatus'].map(employment_map)
    data['PropertyArea'] = data['PropertyArea'].map(property_map)

    # Separate Features and Target
    X = data.drop(['ApplicantID', 'LoanApproved'], axis=1)
    y = data['LoanApproved']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'model': model,
        'scaler': scaler,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cm': cm,
        'df_clean': df_clean,
        'feature_names': X.columns.tolist()
    }


# Train & Cache model
model_data = load_and_train_model()

# --- Matplotlib Plot styling configuration for Light/Red/Gray theme ---
def configure_plot_theme():
    plt.style.use('default')
    plt.rcParams['text.color'] = '#1E293B'
    plt.rcParams['axes.labelcolor'] = '#1E293B'
    plt.rcParams['xtick.color'] = '#475569'
    plt.rcParams['ytick.color'] = '#475569'
    plt.rcParams['figure.facecolor'] = '#FFFFFF'
    plt.rcParams['axes.facecolor'] = '#F8FAFC'
    plt.rcParams['axes.edgecolor'] = '#E2E8F0'
    plt.rcParams['grid.color'] = '#F1F5F9'
    plt.rcParams['font.family'] = 'sans-serif'


configure_plot_theme()

# --- Header section (Institutional Classic Style) ---
st.markdown("""
<div class="title-container">
    <div class="title-header">PRIME TRUST</div>
    <div class="subtitle-header">Institutional Loan Analytics & Risk Assessment Platform</div>
</div>
""", unsafe_allow_html=True)

# --- Define Tabs ---
tab1, tab2 = st.tabs(["🏛️ CREDIT ELIGIBILITY ASSESSMENT", "📊 PORTFOLIO ANALYTICS"])

with tab1:
    st.markdown('<p style="color:#475569; font-size:1rem; text-align:center; margin-bottom: 30px; font-style: italic;">Enter client portfolio attributes below to perform a risk assessment query.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-title">👤 Applicant Demographics</h4>', unsafe_allow_html=True)
        gender = st.selectbox("Client Gender", ["Male", "Female"])
        age = st.slider("Client Age (Years)", 21, 65, 30)
        property_area = st.selectbox("Property Zone", ["Urban", "Semi-Urban", "Rural"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-title">💼 Employment & Liquidity</h4>', unsafe_allow_html=True)
        employment = st.selectbox("Professional Status", ["Employed", "Self-Employed", "Unemployed"])
        monthly_income = st.number_input("Monthly Income ($)", min_value=5000, max_value=250000, value=45000, step=1000)
        existing_loans = st.number_input("Active Credit Lines", min_value=0, max_value=10, value=1, step=1)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h4 class="section-title">📈 Underwriting Terms</h4>', unsafe_allow_html=True)
        loan_amount = st.number_input("Requested Principal ($)", min_value=10000, max_value=5000000, value=250000, step=5000)
        loan_term = st.selectbox("Amortization Term (Months)", [12, 24, 36, 48, 60], index=2)
        credit_score = st.slider("Bureau Credit Score (FICO)", 300, 900, 680)
        st.markdown('</div>', unsafe_allow_html=True)

    # Dynamic calculation of financial metrics
    dti_ratio = (loan_amount / loan_term) / monthly_income if monthly_income > 0 else 0
    dti_percentage = dti_ratio * 100

    st.markdown('<div class="dti-wrapper">', unsafe_allow_html=True)
    st.markdown(f'<h5 style="margin: 0; color:#475569; font-family:\'Inter\', sans-serif;">Calculated Debt-to-Income (DTI) Ratio: <span style="color:{"#B91C1C" if dti_percentage > 40 else "#16A34A"}; font-weight:700;">{dti_percentage:.1f}%</span></h5>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Predict Button Action
    predict_btn = st.button("RUN RISK ASSESSMENT MODEL")

    if predict_btn:
        with st.spinner("Executing Random Forest risk classification..."):
            import time
            time.sleep(1.0)

        # Map user input values exactly to encoder categories
        gender_map = {'Male': 1, 'Female': 0}
        employment_map = {'Employed': 0, 'Self-Employed': 1, 'Unemployed': 2}
        property_map = {'Rural': 0, 'Semi-Urban': 1, 'Urban': 2}

        # Build feature DataFrame matching features order exactly
        user_df = pd.DataFrame([[
            gender_map[gender],
            age,
            monthly_income,
            loan_amount,
            credit_score,
            employment_map[employment],
            existing_loans,
            loan_term,
            property_map[property_area]
        ]], columns=model_data['feature_names'])

        # Scale features
        user_scaled = model_data['scaler'].transform(user_df)

        # Predict probability & final decision
        pred_prob = model_data['model'].predict_proba(user_scaled)[0]
        prediction = model_data['model'].predict(user_scaled)[0]

        approval_probability = pred_prob[1] * 100

        # Beautiful Custom Result Box (Adapted for red & white light theme)
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card approved-card">
                <div class="checkmark-wrapper">
                    <svg class="checkmark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 52 52">
                        <circle class="checkmark__circle" cx="26" cy="26" r="25" fill="none"/>
                        <path class="checkmark__check" fill="none" d="M14.1 27.2l7.1 7.2 16.7-16.8"/>
                    </svg>
                </div>
                <h2 style="color: #16A34A; font-weight: 700; margin-top: 0; margin-bottom: 10px;">Application Approved</h2>
                <h4 style="color: #0F172A; margin-bottom: 20px;">The underwriting model approved the risk profile (Confidence rate: <strong>{approval_probability:.1f}%</strong>)</h4>
                <p style="color: #475569; max-width: 650px; margin: 0 auto; line-height: 1.6; font-size: 0.95rem;">
                    The applicant's FICO Score ({credit_score}) meets the institution's prime borrower standards. Robust monthly income profiles and appropriate debt serviceability metrics justify underwriting approval.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card rejected-card">
                <div class="crossmark-wrapper">
                    <svg class="crossmark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 52 52">
                        <circle class="crossmark__circle" cx="26" cy="26" r="25" fill="none"/>
                        <path class="crossmark__path" fill="none" d="M16 16 36 36 M36 16 16 36"/>
                    </svg>
                </div>
                <h2 style="color: #B91C1C; font-weight: 700; margin-top: 0; margin-bottom: 10px;">Application Declined</h2>
                <h4 style="color: #0F172A; margin-bottom: 20px;">The risk index exceeds tolerable underwriting limits (Confidence rate: <strong>{100 - approval_probability:.1f}%</strong>)</h4>
                <p style="color: #475569; max-width: 650px; margin: 0 auto 20px auto; line-height: 1.6; font-size: 0.95rem;">
                    Structural deficiencies in applicant attributes detected. Risks may arise from weak credit history, high calculated DTI ratios ({dti_percentage:.1f}%), or current unemployment.
                </p>
                <div style="background: #FFF5F5; border-radius: 6px; padding: 16px; max-width: 500px; margin: 0 auto; border: 1px solid #FEE2E2">
                    <strong style="color: #991B1B; font-size: 0.95rem;">Corrective Guidance:</strong>
                    <ul style="color: #475569; text-align: left; margin-top: 8px; font-size: 0.9rem; line-height: 1.4;">
                        <li>Advise client to improve bureau score to 650+ by settling outstanding balances.</li>
                        <li>Suggest loan structuring adjustments (e.g., lower requested principal).</li>
                        <li>Request additional security/co-signers to offset debt repayment risks.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown('<p style="color:#475569; font-size:1rem; text-align:center; margin-bottom: 30px; font-style: italic;">Historical model classification characteristics and portfolio distribution figures.</p>', unsafe_allow_html=True)

    # Show performance metrics row
    st.markdown("""
    <div class="metrics-row">
        <div class="custom-metric">
            <div class="metric-val" style="color: #B91C1C;">{:.2f}%</div>
            <div class="metric-lbl">Model Accuracy</div>
        </div>
        <div class="custom-metric">
            <div class="metric-val" style="color: #334155;">{:.2f}%</div>
            <div class="metric-lbl">Underwriting Precision</div>
        </div>
        <div class="custom-metric">
            <div class="metric-val" style="color: #334155;">{:.2f}%</div>
            <div class="metric-lbl">Recall Sensitivity</div>
        </div>
        <div class="custom-metric">
            <div class="metric-val" style="color: #334155;">{:.2f}%</div>
            <div class="metric-lbl">Weighted F1 Score</div>
        </div>
    </div>
    """.format(
        model_data['accuracy'] * 100,
        model_data['precision'] * 100,
        model_data['recall'] * 100,
        model_data['f1'] * 100
    ), unsafe_allow_html=True)

    df = model_data['df_clean']

    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h5 style="color:#0F172A; text-align:center; margin-bottom:15px; font-weight:600;">💼 Professional Status vs Approval Rate</h5>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(
            x='EmploymentStatus',
            hue='LoanApproved',
            data=df,
            palette={0: '#94A3B8', 1: '#B91C1C'},
            ax=ax
        )
        ax.set_ylabel("Count")
        ax.set_xlabel("Employment Status")
        new_labels = ['Declined', 'Approved']
        for t, l in zip(ax.legend().get_texts(), new_labels):
            t.set_text(l)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_plot2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h5 style="color:#0F172A; text-align:center; margin-bottom:15px; font-weight:600;">📊 Requested Principal vs Liquidity Distribution</h5>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            x='MonthlyIncome',
            y='LoanAmount',
            hue='LoanApproved',
            data=df,
            palette={0: '#94A3B8', 1: '#B91C1C'},
            alpha=0.65,
            ax=ax
        )
        ax.set_xlabel("Monthly Income ($)")
        ax.set_ylabel("Loan Amount ($)")
        new_labels = ['Declined', 'Approved']
        for t, l in zip(ax.legend().get_texts(), new_labels):
            t.set_text(l)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    col_plot3, col_plot4 = st.columns(2)

    with col_plot3:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h5 style="color:#0F172A; text-align:center; margin-bottom:15px; font-weight:600;">🔗 Underwriting Feature Correlations</h5>', unsafe_allow_html=True)

        # Map temporary encoding for numerical correlation visualization
        numeric_df = df.copy()
        gender_map = {'Male': 1, 'Female': 0}
        employment_map = {'Employed': 0, 'Self-Employed': 1, 'Unemployed': 2}
        property_map = {'Rural': 0, 'Semi-Urban': 1, 'Urban': 2}
        numeric_df['Gender'] = numeric_df['Gender'].map(gender_map)
        numeric_df['EmploymentStatus'] = numeric_df['EmploymentStatus'].map(employment_map)
        numeric_df['PropertyArea'] = numeric_df['PropertyArea'].map(property_map)

        numeric_data = numeric_df.select_dtypes(include=np.number).drop('ApplicantID', axis=1, errors='ignore')

        fig, ax = plt.subplots(figsize=(6, 4))
        # Use a classic gray-red-blue diverging colormap (RdGy)
        sns.heatmap(
            numeric_data.corr(),
            annot=True,
            fmt='.2f',
            cmap='RdGy',
            annot_kws={"size": 7},
            ax=ax,
            cbar=False
        )
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_plot4:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h5 style="color:#0F172A; text-align:center; margin-bottom:15px; font-weight:600;">🍕 Total Underwriting Approval Rate</h5>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        approved_counts = df['LoanApproved'].value_counts()
        ax.pie(
            approved_counts,
            labels=['Approved', 'Declined'],
            colors=['#B91C1C', '#E2E8F0'],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'color': '#1E293B'},
            wedgeprops={'edgecolor': '#FFFFFF', 'linewidth': 2}
        )
        ax.axis('equal')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
