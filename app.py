import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import PartialDependenceDisplay

# =========================================================
# TICKET 3.3.1: CONFIG & BASIC STRUCTURE
# =========================================================
st.set_page_config(
    page_title="Flu Vaccination Interactive Dashboard",
    page_icon="💉",
    layout="wide"
)

st.title("Flu Vaccination Interactive Dashboard")

# Implement session state management for persistence
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None
if "last_input" not in st.session_state:
    st.session_state["last_input"] = None
if "model_type" not in st.session_state:
    st.session_state["model_type"] = None

# =========================================================
# DATA AND MODEL LOADING (Optimized for speed)
# =========================================================

@st.cache_data(show_spinner="Loading data...")
def load_data():
    """Loads and caches the flu_df dataset from flu_df.csv."""
    try:
        data = pd.read_csv("flu_df.csv")

        # Ensure a lowercase binned age group column is available for visualizations.
        if "age_group" in data.columns and "age_group_binned" not in data.columns:
            data["age_group_binned"] = pd.cut(
                data["age_group"],
                bins=[18, 40, 60, 85, 100],
                labels=["Young (18-39)", "Middle (40-59)", "Elderly (60-84)", "Senior (85+)"],
                right=False,
                include_lowest=True
            ).astype(str).replace("nan", "Missing")

        return data
    except FileNotFoundError:
        st.error("Data file 'flu_df.csv' not found. Please save it correctly from your notebook.")
        return pd.DataFrame()


@st.cache_resource(show_spinner="Loading best tuned models...")
def load_tuned_models():
    """Loads the best-tuned pipelines for H1N1 and Seasonal (Ticket 3.3.3)."""
    try:
        h1n1_pipe = joblib.load("h1n1_pipeline.pkl")
        seas_pipe = joblib.load("seas_pipeline.pkl")

        h1n1_model_type = type(h1n1_pipe.named_steps["model"]).__name__
        seas_model_type = type(seas_pipe.named_steps["model"]).__name__

        return {
            "h1n1_vaccine": {"pipe": h1n1_pipe, "type": h1n1_model_type},
            "seasonal_vaccine": {"pipe": seas_pipe, "type": seas_model_type},
        }
    except FileNotFoundError:
        st.error("Model files (h1n1_pipeline.pkl or seas_pipeline.pkl) not found. Cannot run predictions.")
        return None


@st.cache_data(show_spinner="Loading feature insights...")
def load_feature_insights():
    """Loads feature importance data and correlation data (Ticket 3.3.4)."""
    fi_df = pd.DataFrame()
    corr_df = pd.DataFrame()

    try:
        fi_df = pd.read_csv("feature_importance.csv")
        fi_df["target"] = fi_df["target"].str.lower()
    except FileNotFoundError:
        st.warning("Feature importance file 'feature_importance.csv' not found.")

    try:
        corr_df = pd.read_csv("correlation_matrix.csv", index_col=0)
    except FileNotFoundError:
        st.warning("Correlation matrix file 'correlation_matrix.csv' not found.")

    return fi_df, corr_df


# Helper function to get preprocessed feature names for SHAP/PDP
def get_processed_feature_names(pipeline):
    preprocessor = pipeline.named_steps.get("preprocess")
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        return preprocessor.get_feature_names_out()
    return None


# =========================================================
# GLOBAL LOAD
# =========================================================
flu_df = load_data()
tuned_models = load_tuned_models()
feature_importance_df, correlation_matrix = load_feature_insights()

# Extract feature and column details
feature_cols, num_cols, cat_cols = [], [], []
if not flu_df.empty:
    X_sample = flu_df.drop(
        columns=[c for c in flu_df.columns if "vaccine" in c.lower() or "respondent_id" in c.lower()],
        errors="ignore"
    )
    feature_cols = X_sample.columns.tolist()
    num_cols = X_sample.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]


# =========================================================
# REUSABLE COMPONENT (TICKET 3.3.1 & 3.3.2)
# =========================================================

def plot_vaccination_rate_by(flu_df, target_col, group_col, title):
    """Reusable function to plot rates by a categorical group."""
    if flu_df.empty or group_col not in flu_df.columns or target_col not in flu_df.columns:
        st.info(f"Cannot plot: Missing column '{group_col}' or data is empty.")
        return

    grouped = (
        flu_df.groupby(group_col)[target_col]
        .mean()
        .reset_index()
        .rename(columns={target_col: "vaccination_rate"})
    )
    grouped["vaccination_rate"] = grouped["vaccination_rate"] * 100

    fig = px.bar(
        grouped,
        x=group_col,
        y="vaccination_rate",
        title=title,
        labels={"vaccination_rate": "Acceptance Rate (%)"},
    )
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to section:",
    [
        "Prediction Interface",
        "Feature Exploration",
        "Demographic Dashboard",
        "Insights & Recommendations",
    ],
)


# =========================================================
# PAGE: PREDICTION INTERFACE (TICKET 3.3.3)
# =========================================================
if page == "Prediction Interface":
    st.header("Interactive Prediction & Explanation Tool")

    if not tuned_models or flu_df.empty:
        st.stop()

    target_choice = st.radio(
        "Which vaccine do you want to predict?",
        options=["h1n1_vaccine", "seasonal_vaccine"],
        format_func=lambda x: "H1N1 Vaccine Acceptance" if "h1n1" in x.lower() else "Seasonal Vaccine Acceptance",
        horizontal=True,
    )

    model_info = tuned_models[target_choice]
    model = model_info["pipe"]

    st.markdown("### Build an interactive form for entering predictor variables")

    with st.form("prediction_form"):
        colA, colB = st.columns(2)
        input_data = {}

        # Numeric features
        for col in num_cols:
            with colA:
                s = flu_df[col].dropna()
                val = st.slider(
                    col.replace("_", " ").title(),
                    float(s.min()),
                    float(s.max()),
                    float(s.median()),
                    help=f"Select the value for {col}."
                )
                input_data[col] = val

        # Categorical features
        for col in cat_cols:
            with colB:
                s = flu_df[col].dropna().astype(str)
                opts = sorted(s.unique().tolist())
                default_val = str(s.mode().iloc[0]) if len(s.mode()) > 0 else opts[0]
                val = st.selectbox(
                    col.replace("_", " ").title(),
                    options=opts,
                    index=opts.index(default_val) if default_val in opts else 0,
                    help=f"Select the category for {col}."
                )
                input_data[col] = val

        predict_button = st.form_submit_button("Predict Probability and Explain")

    if predict_button:
        X_input = pd.DataFrame([input_data])
        prob = model.predict_proba(X_input)[0, 1]
        st.session_state["last_prediction"] = prob
        st.session_state["last_input"] = X_input

    if st.session_state["last_prediction"] is not None:
        prob = st.session_state["last_prediction"]
        X_input = st.session_state["last_input"]

        st.subheader("Prediction Result")

        col_p, col_c = st.columns(2)
        with col_p:
            st.metric(
                "Predicted Acceptance Probability",
                f"{prob:.1%}",
                help="This is the model's computed probability."
            )

        with col_c:
            confidence = abs(prob - 0.5) * 2
            st.metric(
                "Model Confidence (0-1)",
                f"{confidence:.2f}",
                help="Confidence measures the prediction's distance from 50% uncertainty."
            )
            st.progress(float(np.clip(confidence, 0, 1)))

        st.markdown("---")

        st.subheader("Local Explanation: How Inputs Affect the Prediction")
        st.caption(
            "The **SHAP** chart shows which input variables pushed the prediction "
            "probability away from the model's average."
        )

        try:
            processed_feature_names = get_processed_feature_names(model)
            X_input_processed = model.named_steps["preprocess"].transform(X_input)

            X_train_sampled = flu_df[feature_cols].sample(100, random_state=42)
            X_train_processed = model.named_steps["preprocess"].transform(X_train_sampled)

            explainer = shap.TreeExplainer(model.named_steps["model"], X_train_processed)
            shap_values = explainer.shap_values(X_input_processed)

            if isinstance(shap_values, list):
                shap_vals = shap_values[1].flatten()
            else:
                shap_vals = shap_values.flatten()

            shap_df = pd.DataFrame(
                {
                    "feature": processed_feature_names,
                    "shap_value": shap_vals
                }
            )

            top_shap = (
                shap_df.assign(abs_shap=lambda x: x["shap_value"].abs())
                .sort_values(by="abs_shap", ascending=False)
                .head(10)
            )

            fig_shap = px.bar(
                top_shap,
                x="shap_value",
                y="feature",
                orientation="h",
                title="Top 10 Feature Contributions to This Prediction",
                color="shap_value",
                color_continuous_scale=px.colors.diverging.RdBu,
                labels={"shap_value": f"SHAP Value (Impact on {target_choice.split('_')[0].upper()})"}
            )
            fig_shap.update_layout(yaxis={"categoryorder": "total ascending"})

            st.plotly_chart(fig_shap, use_container_width=True)

        except Exception as e:
            st.error(f"Could not generate SHAP explanation. (Error: {e})")


# =========================================================
# PAGE: FEATURE EXPLORATION (TICKET 3.3.4)
# =========================================================
elif page == "Feature Exploration":
    st.header("Feature Exploration: Model Explanations")

    st.markdown("### Global Feature Importance (Top 10)")
    st.caption(
        "These charts show the overall top 10 most impactful features across "
        "the entire dataset for each vaccine model."
    )

    if not feature_importance_df.empty:
        targets = feature_importance_df["target"].unique()
        col_h1n1, col_seas = st.columns(2)

        for target in targets:
            df_target = feature_importance_df[feature_importance_df["target"] == target].copy()
            top_features = df_target.sort_values("abs_importance", ascending=False).head(10)

            fig_fi = px.bar(
                top_features,
                x="importance",
                y="feature",
                orientation="h",
                title=f"Top 10 Features: {target.upper()} Model",
                color="importance",
                color_continuous_scale=px.colors.diverging.RdBu,
                labels={"importance": "Feature Importance Score"},
                height=400
            )
            fig_fi.update_layout(yaxis={"categoryorder": "total ascending"})

            if "h1n1" in target:
                with col_h1n1:
                    st.plotly_chart(fig_fi, use_container_width=True)
            else:
                with col_seas:
                    st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.error("Feature importance data ('feature_importance.csv') not loaded.")

    st.markdown("---")

    st.markdown("### Partial Dependence Plot (PDP) Explorer")
    st.caption(
        "PDPs show the marginal effect of a feature on the predicted probability. "
        "This helps visualize non-linear feature impacts."
    )

    if tuned_models and not flu_df.empty:
        pdp_target = st.radio(
            "Select PDP Target:",
            ["h1n1_vaccine", "seasonal_vaccine"],
            key="pdp_target",
            horizontal=True
        )

        pdp_model = tuned_models[pdp_target]["pipe"]
        top_fi_names = feature_importance_df["feature"].unique().tolist()

        pdp_feature = st.selectbox(
            "Select Feature for PDP:",
            options=top_fi_names,
            help="Select a feature to see how its value changes the prediction probability."
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        try:
            X_pdp = flu_df[feature_cols].sample(500, random_state=42)

            PartialDependenceDisplay.from_estimator(
                pdp_model,
                X_pdp,
                features=[pdp_feature],
                target=1,
                ax=ax,
                feature_names=feature_cols,
                line_kw={"label": "Prediction Probability", "color": "red"}
            )
            ax.set_title(f"PDP for {pdp_feature.replace('_', ' ').title()}")

            st.pyplot(fig)
            st.caption(
                f"The line shows how the likelihood of vaccine acceptance changes as "
                f"the value of '{pdp_feature}' changes."
            )

        except Exception as e:
            st.error(f"Could not generate Partial Dependence Plot. (Error: {e})")


# =========================================================
# PAGE: DEMOGRAPHIC DASHBOARD (TICKET 3.3.2 & 3.3.4 Comparison)
# =========================================================
elif page == "Demographic Dashboard":
    st.header("Demographic Trends & Vaccination Patterns")

    st.markdown("### Vaccination Rates by Key Demographics (Comparison Views)")
    col1, col2 = st.columns(2)

    with col1:
        if "age_group_binned" in flu_df.columns:
            group_col = "age_group_binned"
        elif "age_group" in flu_df.columns:
            group_col = "age_group"
        else:
            group_col = "education"

        if group_col in flu_df.columns:
            plot_vaccination_rate_by(
                flu_df,
                "h1n1_vaccine",
                group_col,
                f"H1N1 Acceptance Rate by {group_col.replace('_', ' ').title()}"
            )

    with col2:
        plot_vaccination_rate_by(
            flu_df,
            "seasonal_vaccine",
            "education",
            "Seasonal Acceptance Rate by Education"
        )

    st.markdown("---")

    st.markdown("### Correlation Analysis Visualization (Numeric Features)")
    if not correlation_matrix.empty:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap of Numeric Variables")
        st.pyplot(fig)
        st.caption(
            "A score near +1 indicates a strong positive relationship; "
            "near -1 is a strong inverse relationship."
        )
    else:
        st.warning("Correlation matrix data not available. Ensure 'correlation_matrix.csv' is saved.")


# =========================================================
# PAGE: INSIGHTS & RECOMMENDATIONS (TICKET 3.3.5)
# =========================================================
elif page == "Insights & Recommendations":
    st.header("Actionable Insights & Strategic Recommendations")

    st.subheader("Key Findings (Supported by Data)")
    if not feature_importance_df.empty:
        st.subheader("Visual Summary: Top Predictors")

        df_seas = feature_importance_df[
            feature_importance_df["target"].str.contains("seasonal", na=False)
        ].copy()

        top_seas = df_seas.sort_values("abs_importance", ascending=False).head(5)

        fig_summary = px.bar(
            top_seas,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 5 Drivers of Seasonal Flu Acceptance (Visual Evidence)",
            color="importance",
            color_continuous_scale=px.colors.diverging.RdBu,
            height=300
        )
        fig_summary.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_summary, use_container_width=True)

    st.markdown(
        """
        Based on model analysis, **Doctor Recommendation**, **Concern Level (Risk Perception)**,
        and **Effectiveness Opinion** are the most critical factors influencing a patient's
        decision for both vaccines.
        """
    )

    st.markdown("---")

    st.subheader("Tailored Public Health Recommendations")
    st.caption("Select a demographic group to view targeted outreach strategies.")

    demo_groups = [
        "Overall Strategy",
        "Low Income Households",
        "Seniors (60+)",
        "Parents of Young Children",
        "Low Education"
    ]
    selected_group = st.selectbox("Select Target Audience:", demo_groups)

    recos = {
        "Overall Strategy": [
            "**Leverage Healthcare Providers:** Ensure all primary care physicians give strong, personalized vaccine recommendations.",
            "**Focus on Perceived Risk:** Run campaigns that communicate personal and community risk using clear, evidence-based messages.",
            "**Simplify Access:** Promote easy access points (pharmacies, workplace clinics) to lower barriers to uptake."
        ],
        "Low Income Households": [
            "**Address Cost/Access Barriers:** Provide free or subsidized vaccination at neighborhood clinics and community centers.",
            "**Use Trusted Messengers:** Partner with local community leaders and non-profit organizations for outreach.",
            "**Highlight Economic Benefits:** Emphasize fewer sick days and reduced out-of-pocket costs from avoiding illness."
        ],
        "Seniors (60+)": [
            "**Targeted Doctor Outreach:** Flag patients 60+ for follow-up reminders if they have not received the vaccine.",
            "**Focus on Co-morbidities:** Communicate the increased risk severity due to age and existing conditions.",
            "**On-site Clinics:** Offer vaccination at senior centers and assisted living facilities."
        ],
        "Parents of Young Children": [
            "**Address Household Spread:** Highlight the risk of transmitting flu to children and other family members.",
            "**Co-vaccination Opportunities:** Encourage parents to vaccinate when children receive routine shots.",
            "**School-Based Messaging:** Use school newsletters, PTA groups, and pediatric clinics for communication."
        ],
        "Low Education": [
            "**Use Clear, Simple Language:** Rely on visuals and avoid complex medical terms in communication materials.",
            "**Face-to-Face Consultations:** Provide opportunities for in-person Q&A with healthcare professionals.",
            "**Build Trust Over Time:** Use consistent, repeated messaging from trusted local providers and organizations."
        ]
    }

    st.markdown("### Recommended Actions for Selected Group")
    for item in recos[selected_group]:
        st.markdown(f"- {item}")
