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
    page_title="PrediLoan | Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Styling & CSS Animations ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* Main font styling */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif !important;
}

/* Glassmorphism containers */
.glass-container {
    background: rgba(30, 41, 59, 0.45);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 24px;
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

.title-container {
    background: linear-gradient(135deg, #1E1B4B 0%, #0F172A 100%);
    border: 1px solid rgba(79, 70, 229, 0.2);
    border-radius: 16px;
    padding: 30px;
    margin-bottom: 25px;
    text-align: center;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

.title-header {
    background: linear-gradient(90deg, #818CF8, #C084FC);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.8rem;
    margin-bottom: 10px;
}

.subtitle-header {
    color: #94A3B8;
    font-size: 1.1rem;
    font-weight: 300;
}

/* Tab Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
    justify-content: center;
    border-bottom: 1px solid #1E293B;
}

.stTabs [data-baseweb="tab"] {
    background-color: #1E293B;
    border-radius: 10px 10px 0px 0px;
    padding: 12px 28px;
    color: #94A3B8;
    border: 1px solid #334155;
    border-bottom: none;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    font-weight: 500;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #F8FAFC;
    background-color: #334155;
    border-color: #4F46E5;
}

.stTabs [aria-selected="true"] {
    background-color: #4F46E5 !important;
    color: #FFFFFF !important;
    border-color: #4F46E5 !important;
    box-shadow: 0 -4px 10px rgba(79, 70, 229, 0.2);
}

/* Custom Metrics Design */
.metrics-row {
    display: flex;
    gap: 16px;
    margin-bottom: 24px;
}

.custom-metric {
    flex: 1;
    background: rgba(30, 41, 59, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}

.custom-metric:hover {
    transform: translateY(-4px);
    border-color: rgba(79, 70, 229, 0.5);
    box-shadow: 0 4px 20px rgba(79, 70, 229, 0.15);
}

.metric-val {
    font-size: 2.2rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 5px 0;
}

.metric-lbl {
    font-size: 0.9rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Prediction Result Cards */
.result-card {
    border-radius: 16px;
    padding: 30px;
    text-align: center;
    margin-top: 25px;
    border: 1px solid;
    animation: slideUp 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
}

@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.approved-card {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(4, 120, 87, 0.05) 100%);
    border-color: rgba(16, 185, 129, 0.3);
}

.rejected-card {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(185, 28, 28, 0.05) 100%);
    border-color: rgba(239, 68, 68, 0.3);
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
    box-shadow: inset 0px 0px 0px rgba(16, 185, 129, 0.2);
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
    box-shadow: inset 0px 0px 0px rgba(239, 68, 68, 0.2);
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
    50% { transform: scale(1.1); }
}
@keyframes fill-green {
    100% { box-shadow: inset 0px 0px 0px 45px rgba(16, 185, 129, 0.15); }
}
@keyframes fill-red {
    100% { box-shadow: inset 0px 0px 0px 45px rgba(239, 68, 68, 0.15); }
}

.stButton>button {
    background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%) !important;
    color: white !important;
    border: none !important;
    padding: 12px 30px !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
    width: 100%;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5) !important;
}

.stButton>button:active {
    transform: translateY(1px) !important;
}

</style>
""", unsafe_allow_html=True)


# --- Helper Function: Generate Dataset if missing ---
def verify_dataset():
    dataset_path = "loan_dataset.csv"
    if not os.path.exists(dataset_path):
        # Generate synthetic data exactly like loan_approval_prediction.py
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

        # Inject some artificial NaNs for compatibility
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

    # Save a copy of clean data for plotting prior to encoding
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

# --- Matplotlib Plot styling configuration ---
def configure_plot_theme():
    # Make plots look like part of the dark UI
    plt.style.use('dark_background')
    plt.rcParams['text.color'] = '#94A3B8'
    plt.rcParams['axes.labelcolor'] = '#94A3B8'
    plt.rcParams['xtick.color'] = '#94A3B8'
    plt.rcParams['ytick.color'] = '#94A3B8'
    plt.rcParams['figure.facecolor'] = '#1E293B'
    plt.rcParams['axes.facecolor'] = '#1E293B'
    plt.rcParams['axes.edgecolor'] = '#334155'
    plt.rcParams['grid.color'] = '#334155'


configure_plot_theme()

# --- Header section ---
st.markdown("""
<div class="title-container">
    <div class="title-header">🏦 PrediLoan AI</div>
    <div class="subtitle-header">Next-Generation Loan Decision System powered by Machine Learning</div>
</div>
""", unsafe_allow_html=True)

# --- Define Tabs ---
tab1, tab2 = st.tabs(["⚡ SMART ELIGIBILITY PREDICTOR", "📊 DATASET & INSIGHTS DASHBOARD"])

with tab1:
    st.markdown('<p style="color:#94A3B8; font-size:1.1rem; text-align:center; margin-bottom: 30px;">Input the applicant details below to analyze loan approval probability instantly.</p>', unsafe_allow_html=True)

    # Wrap inputs inside grid layouts
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#818CF8; margin-top:0;">👤 Personal Profile</h4>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age (Years)", 21, 65, 30)
        property_area = st.selectbox("Property Area", ["Urban", "Semi-Urban", "Rural"])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#818CF8; margin-top:0;">💼 Employment & Income</h4>', unsafe_allow_html=True)
        employment = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
        monthly_income = st.number_input("Monthly Income ($)", min_value=5000, max_value=250000, value=45000, step=1000)
        existing_loans = st.number_input("Existing Loans Count", min_value=0, max_value=10, value=1, step=1)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h4 style="color:#818CF8; margin-top:0;">📈 Loan Specifications</h4>', unsafe_allow_html=True)
        loan_amount = st.number_input("Requested Loan Amount ($)", min_value=10000, max_value=5000000, value=250000, step=5000)
        loan_term = st.selectbox("Loan Term (Months)", [12, 24, 36, 48, 60], index=2)
        credit_score = st.slider("Credit Score", 300, 900, 680)
        st.markdown('</div>', unsafe_allow_html=True)

    # Dynamic calculation of financial metrics
    dti_ratio = (loan_amount / loan_term) / monthly_income if monthly_income > 0 else 0
    dti_percentage = dti_ratio * 100

    st.markdown('<div class="glass-container" style="text-align: center;">', unsafe_allow_html=True)
    st.markdown(f'<h5 style="margin: 0; color:#94A3B8;">Estimated Debt-to-Income (DTI) Monthly Ratio: <span style="color:{"#EF4444" if dti_percentage > 40 else "#10B981"}; font-weight:700;">{dti_percentage:.1f}%</span></h5>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Predict Button Action
    predict_btn = st.button("ASSESS LOAN ELIGIBILITY")

    if predict_btn:
        # Progress motion
        with st.spinner("Processing applicant profile through Random Forest classifier..."):
            import time
            time.sleep(1.2) # Elegant delay for UI experience

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

        # Beautiful Custom Result Box
        if prediction == 1:
            st.markdown(f"""
            <div class="result-card approved-card">
                <div class="checkmark-wrapper">
                    <svg class="checkmark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 52 52">
                        <circle class="checkmark__circle" cx="26" cy="26" r="25" fill="none"/>
                        <path class="checkmark__check" fill="none" d="M14.1 27.2l7.1 7.2 16.7-16.8"/>
                    </svg>
                </div>
                <h2 style="color: #10B981; font-weight: 700; margin-bottom: 10px;">Loan Approved!</h2>
                <h4 style="color: #F8FAFC; margin-bottom: 20px;">The algorithm determined the application is eligible with a confidence rate of <strong>{approval_probability:.1f}%</strong></h4>
                <p style="color: #94A3B8; max-width: 600px; margin: 0 auto; line-height: 1.5;">
                    The applicant exhibits a strong credit score ({credit_score}), healthy monthly income compared to the loan request, and a low loan repayment risk profile. Safe to proceed with normal processing.
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
                <h2 style="color: #EF4444; font-weight: 700; margin-bottom: 10px;">Loan Rejected</h2>
                <h4 style="color: #F8FAFC; margin-bottom: 20px;">The algorithm predicts a high probability of defaults (Confidence: <strong>{100 - approval_probability:.1f}%</strong>)</h4>
                <p style="color: #94A3B8; max-width: 600px; margin: 0 auto 15px auto; line-height: 1.5;">
                    The applicant profile carries structural risks. Potential issues might include low credit history (Credit Score: {credit_score}), high debt-to-income ratio ({dti_percentage:.1f}%), or unemployment status.
                </p>
                <div style="background: rgba(239, 68, 68, 0.08); border-radius: 8px; padding: 12px; max-width: 500px; margin: 0 auto; border: 1px solid rgba(239, 68, 68, 0.2)">
                    <strong style="color: #F8FAFC;">Recommended Actions:</strong>
                    <ul style="color: #94A3B8; text-align: left; margin-top: 8px; font-size: 0.95rem;">
                        <li>Review and improve the Credit Score above 650.</li>
                        <li>Consider requesting a lower Loan Amount to improve Debt-to-Income ratio.</li>
                        <li>Consider adding a co-signer or checking the applicant's existing active loans count.</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown('<p style="color:#94A3B8; font-size:1.1rem; text-align:center; margin-bottom: 30px;">Overview of the training dataset, key insights, and Random Forest classifier performance.</p>', unsafe_allow_html=True)

    # Show performance metrics row
    st.markdown("""
    <div class="metrics-row">
        <div class="custom-metric">
            <div class="metric-val" style="color: #818CF8;">{:.2f}%</div>
            <div class="metric-lbl">Model Accuracy</div>
        </div>
        <div class="custom-metric">
            <div class="metric-val" style="color: #34D399;">{:.2f}%</div>
            <div class="metric-lbl">Precision (Approved)</div>
        </div>
        <div class="custom-metric">
            <div class="metric-val" style="color: #FB7185;">{:.2f}%</div>
            <div class="metric-lbl">Recall (Approved)</div>
        </div>
        <div class="custom-metric">
            <div class="metric-val" style="color: #FBBF24;">{:.2f}%</div>
            <div class="metric-lbl">F1 Score</div>
        </div>
    </div>
    """.format(
        model_data['accuracy'] * 100,
        model_data['precision'] * 100,
        model_data['recall'] * 100,
        model_data['f1'] * 100
    ), unsafe_allow_html=True)

    # Plots Section
    df = model_data['df_clean']

    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h5 style="color:#F8FAFC; text-align:center; margin-bottom:15px;">💼 Employment Status vs Approval Rate</h5>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(
            x='EmploymentStatus',
            hue='LoanApproved',
            data=df,
            palette={0: '#EF4444', 1: '#10B981'},
            ax=ax
        )
        ax.set_ylabel("Count")
        ax.set_xlabel("Employment Status")
        # Rename legend labels
        new_labels = ['Rejected', 'Approved']
        for t, l in zip(ax.legend().get_texts(), new_labels):
            t.set_text(l)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_plot2:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h5 style="color:#F8FAFC; text-align:center; margin-bottom:15px;">📊 Income vs Loan Amount Distribution</h5>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            x='MonthlyIncome',
            y='LoanAmount',
            hue='LoanApproved',
            data=df,
            palette={0: '#EF4444', 1: '#10B981'},
            alpha=0.6,
            ax=ax
        )
        ax.set_xlabel("Monthly Income ($)")
        ax.set_ylabel("Loan Amount ($)")
        # Rename legend labels
        new_labels = ['Rejected', 'Approved']
        for t, l in zip(ax.legend().get_texts(), new_labels):
            t.set_text(l)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    col_plot3, col_plot4 = st.columns(2)

    with col_plot3:
        st.markdown('<div class="glass-container">', unsafe_allow_html=True)
        st.markdown('<h5 style="color:#F8FAFC; text-align:center; margin-bottom:15px;">🔗 Feature Correlation Matrix</h5>', unsafe_allow_html=True)

        # We need to map copy to visualize numerical correlation
        numeric_df = df.copy()
        gender_map = {'Male': 1, 'Female': 0}
        employment_map = {'Employed': 0, 'Self-Employed': 1, 'Unemployed': 2}
        property_map = {'Rural': 0, 'Semi-Urban': 1, 'Urban': 2}
        numeric_df['Gender'] = numeric_df['Gender'].map(gender_map)
        numeric_df['EmploymentStatus'] = numeric_df['EmploymentStatus'].map(employment_map)
        numeric_df['PropertyArea'] = numeric_df['PropertyArea'].map(property_map)

        numeric_data = numeric_df.select_dtypes(include=np.number).drop('ApplicantID', axis=1, errors='ignore')

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            numeric_data.corr(),
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
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
        st.markdown('<h5 style="color:#F8FAFC; text-align:center; margin-bottom:15px;">🍕 Overall Loan Approval Distribution</h5>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        approved_counts = df['LoanApproved'].value_counts()
        ax.pie(
            approved_counts,
            labels=['Approved', 'Rejected'],
            colors=['#10B981', '#EF4444'],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'color': '#F8FAFC'},
            wedgeprops={'edgecolor': '#1E293B', 'linewidth': 2}
        )
        # Ensure pie is a circle
        ax.axis('equal')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
