import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- 1. PAGE CONFIG ---
st.set_page_config(
    page_title="Telecom Churn & LTV Dashboard",
    page_icon="📶",
    layout="wide"
)

st.title("Telecom Customer Churn, LTV & Retention Prediction")
st.markdown(
    """
    This dashboard predicts customer churn probability, expected lifetime value (LTV),
    and assigns each customer to a behavioral segment.
    It also provides data-driven insights for business decision-making.
    """
)

# --- 2. LOAD MODELS, SCALERS, AND DATA ---
@st.cache_resource
def load_models_and_data():
    churn_model = joblib.load('classification_model.joblib')
    regression_model = joblib.load('regression_model.joblib')
    cluster_model = joblib.load('cluster_model.joblib')

    scaler_class = joblib.load('classification_scaler.joblib')
    scaler_reg = joblib.load('regression_scaler.joblib')
    scaler_cluster = joblib.load('cluster_scaler.joblib')

    df = pd.read_csv('customer_data_with_personas.csv')

    # Ensure correct column order per model
    class_features = [col for col in df.columns if col not in ['Churn', 'Cluster', 'Persona']]
    reg_features = [col for col in df.columns if col not in ['TotalCharges', 'Cluster', 'Persona']]

    persona_map = {
        1: 'Loyal Champion',
        3: 'High-Value, At-Risk',
        0: 'New & Churn-Prone',
        2: 'Loyal Saver'
    }

    return (
        churn_model, regression_model, cluster_model,
        scaler_class, scaler_reg, scaler_cluster,
        df, class_features, reg_features, persona_map
    )


(
    churn_model, regression_model, cluster_model,
    scaler_class, scaler_reg, scaler_cluster,
    df, CLASS_FEATURES, REG_FEATURES, persona_map
) = load_models_and_data()


# --- 3. HELPER FUNCTIONS ---

def preprocess_input(user_inputs, all_features):
    """Convert user inputs into model-ready format."""
    input_df = pd.DataFrame([user_inputs])

    # Map binary values
    binary_map = {'Yes': 1, 'No': 0, 'Female': 0, 'Male': 1}
    for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        if col in input_df.columns:
            input_df[col] = input_df[col].map(binary_map)

    # Encode categorical columns
    categorical_cols = input_df.select_dtypes(include='object').columns
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True, dtype=int)

    # Align columns with training set
    for col in all_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[all_features]
    return input_df


def metric_card(title, value, color):
    """Custom metric card component."""
    return f"""
    <div style="
        background-color:{color};
        border-radius:10px;
        padding:18px;
        text-align:center;
        color:white;
        box-shadow:0 4px 10px rgba(0,0,0,0.15);
    ">
        <h4 style="margin:0;font-size:1.05em;">{title}</h4>
        <h2 style="margin:0;font-size:1.8em;">{value}</h2>
    </div>
    """


# --- 4. USER INPUT FORM ---
with st.expander("Customer Details", expanded=True):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        Partner = st.selectbox("Partner", ["No", "Yes"])
        Dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (Months)", 0, 72, 12)

    with col2:
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col3:
        PaymentMethod = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
        MonthlyCharges = st.slider("Monthly Charges ($)", 18, 120, 70)
        TotalCharges = st.slider("Total Charges ($)", 0, 8000, 2000)

    predict_button = st.button("Predict Customer Outcome", type="primary", use_container_width=True)


# --- 5. PREDICTIONS ---
if predict_button:
    user_inputs = {
        'gender': gender,
        'SeniorCitizen': 1 if SeniorCitizen == 'Yes' else 0,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'TechSupport': TechSupport,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    # Merge feature sets for preprocessing
    full_input_df = preprocess_input(user_inputs, list(set(CLASS_FEATURES + REG_FEATURES)))

    # --- CHURN PREDICTION ---
    class_input = full_input_df.copy()[CLASS_FEATURES]
    class_input[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler_class.transform(
        class_input[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )

    churn_prob = churn_model.predict_proba(class_input)[0][1]
    churn_prob_pct = f"{churn_prob * 100:.1f}%"

    # --- LTV PREDICTION ---
    reg_input = full_input_df.copy()[REG_FEATURES]
    reg_input['Churn'] = int(churn_prob > 0.5)
    reg_input = reg_input[REG_FEATURES]  # maintain order
    reg_input[['tenure', 'MonthlyCharges']] = scaler_reg.transform(
        reg_input[['tenure', 'MonthlyCharges']]
    )

    ltv = regression_model.predict(reg_input)[0]
    ltv_formatted = f"${ltv:,.2f}"

    # --- CUSTOMER SEGMENTATION ---
    cluster_input = full_input_df[['tenure', 'MonthlyCharges', 'TotalCharges']].copy()
    cluster_input = scaler_cluster.transform(cluster_input)
    cluster_label = cluster_model.predict(cluster_input)[0]
    persona = persona_map.get(cluster_label, "Unknown Segment")

    # --- METRICS ---
    st.markdown("---")
    st.subheader("Prediction Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        color = "#d9534f" if churn_prob > 0.5 else "#5cb85c"
        st.markdown(metric_card("Churn Probability", churn_prob_pct, color), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Predicted Lifetime Value", ltv_formatted, "#0275d8"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("Customer Segment", persona, "#f0ad4e"), unsafe_allow_html=True)

    # --- 6. INSIGHTS ---
    st.markdown("---")
    st.subheader("Data Insights")

    col_a, col_b = st.columns(2)

    # Insight 1: Churn rate by contract type
    with col_a:
        contract_churn = df.groupby('Contract_One year')['Churn'].mean().reset_index()
        contract_churn.columns = ['Contract Type (One Year = 1)', 'Avg Churn Rate']
        fig1 = px.bar(
            contract_churn,
            x='Contract Type (One Year = 1)',
            y='Avg Churn Rate',
            title="Churn Rate by Contract Type",
            color='Avg Churn Rate',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig1, use_container_width=True)

    # Insight 2: Monthly charges vs churn
    with col_b:
        fig2 = px.box(
            df,
            x='Churn',
            y='MonthlyCharges',
            title="Monthly Charges vs. Churn Status",
            color='Churn',
            color_discrete_sequence=['#5cb85c', '#d9534f']
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Insight 3: Tenure vs churn scatter
    col_c, col_d = st.columns(2)
    with col_c:
        fig3 = px.scatter(
            df.sample(min(2000, len(df))),  # smaller sample for speed
            x='tenure',
            y='MonthlyCharges',
            color='Churn',
            title="Tenure vs Monthly Charges (Churn Patterns)",
            color_discrete_sequence=['#5cb85c', '#d9534f']
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Insight 4: Persona-based revenue distribution
    with col_d:
        persona_rev = df.groupby('Persona')['TotalCharges'].mean().reset_index()
        fig4 = px.bar(
            persona_rev,
            x='Persona',
            y='TotalCharges',
            title="Average Revenue per Customer Segment",
            color='TotalCharges',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig4, use_container_width=True)

else:
    st.info("Fill out the form above and click **Predict Customer Outcome** to view predictions and insights.")
