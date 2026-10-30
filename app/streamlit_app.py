
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config as C
from src import predict, viz

st.set_page_config(page_title="Early Disease Prediction", page_icon="🩺", layout="wide")
viz.set_style()


# ---- cached loading ------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the model ...")
def get_model():
    return predict.load_model()


@st.cache_data
def get_meta():
    return predict.load_metadata()


def figure_path(name: str):
    path = C.FIG_DIR / f"{name}.png"
    return str(path) if path.exists() else None


model, meta = get_model(), get_meta()

# ---- header --------------------------------------------------------------------
st.title("🩺 Early Disease Prediction")
st.caption("Early Disease Prediction Using Machine Learning on Health Data")
st.warning("**Not a medical tool.** " + C.DISCLAIMER)

with st.sidebar:
    st.header("About the model")
    st.write(f"**Model:** {meta['best_model']}")
    test = meta["best_model_test_at_default_threshold"]
    st.write(f"**Test ROC-AUC:** {test['roc_auc']:.3f}")
    st.write(f"**Screening threshold:** {meta['screening_threshold']:.2f}")
    st.write(f"**Trained on:** {meta['data']['n_train']:,} rows")
    st.caption("The data is partly synthetic. See the tab 'Data, ethics and limits'.")

tab_risk, tab_perf, tab_about = st.tabs(["Risk check", "Model performance", "Data, ethics and limits"])

# ---- tab 1: risk check -----------------------------------------------------------
with tab_risk:
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.subheader("Patient values")
        with st.form("patient_form"):
            age = st.slider("Age (years)", 25, 75, 50, key="age")
            bmi = st.slider("BMI (kg/m²)", 18.0, 40.0, 28.0, 0.1, key="bmi")
            blood_pressure = st.slider("Blood pressure (mmHg)", 60, 140, 100, key="blood_pressure")
            glucose = st.slider("Glucose (mg/dL)", 70, 200, 120, key="glucose")
            cholesterol = st.slider("Cholesterol (mg/dL)", 150, 280, 210, key="cholesterol")
            smoker = st.radio("Smoker?", ["No", "Yes"], horizontal=True, key="smoker")
            submitted = st.form_submit_button("Estimate risk", type="primary")
        st.caption("Values are limited to the ranges the model was trained on (25-75 years, BMI 18-40, and so on).")

    if submitted:
        patient = {"age": age, "bmi": bmi, "blood_pressure": blood_pressure, "glucose": glucose,
                   "cholesterol": cholesterol, "smoking": 1 if smoker == "Yes" else 0}
        try:
            st.session_state["result"] = {
                "patient": patient,
                "prediction": predict.predict_patient(patient, model=model, meta=meta),
                "contrib": predict.explain_patient(patient, model=model),
            }
        except ValueError as err:
            st.session_state.pop("result", None)
            right.error(f"Please check the inputs: {err}")

    with right:
        st.subheader("Result")
        res = st.session_state.get("result")
        if res is None:
            st.info("Set the values on the left and press **Estimate risk**.")
        else:
            pred, contrib = res["prediction"], res["contrib"]
            prob = pred["probability"]
            c1, c2 = st.columns(2)
            c1.metric("Estimated probability of disease", f"{prob:.0%}")
            c2.metric("Risk band", pred["risk_band"])

            if pred["risk_band"] == "Lower risk":
                st.success(f"Below the screening threshold ({pred['screening_threshold']:.2f}).")
            elif pred["risk_band"] == "Elevated risk":
                st.warning(f"Above the screening threshold ({pred['screening_threshold']:.2f}). "
                           "A follow-up check would be suggested in a real screening programme.")
            else:
                st.error("High estimated risk. In a real screening programme this would prompt a clinical follow-up.")

            st.markdown("**What moved this estimate** (percentage points, compared with the average reference patient)")
            base = contrib.attrs["base_probability"]
            plot_df = contrib.iloc[::-1]
            fig, ax = plt.subplots(figsize=(6.4, 3.4))
            colors = [viz.RED if v > 0 else viz.BLUE for v in plot_df["contribution_pp"]]
            ax.barh(plot_df["feature"], plot_df["contribution_pp"], color=colors)
            ax.axvline(0, color="black", lw=0.8)
            for y, v in enumerate(plot_df["contribution_pp"]):
                ax.text(v + (0.4 if v >= 0 else -0.4), y, f"{v:+.1f}", va="center",
                        ha="left" if v >= 0 else "right", fontsize=9)
            ax.set(xlabel="Change in estimated risk (percentage points)")
            lim = max(abs(plot_df["contribution_pp"])) * 1.25 + 1
            ax.set_xlim(-lim, lim)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"Average reference risk {base:.0%} + the contributions above = this patient's estimate of {prob:.0%}. "
                       "Red bars raise the estimate, blue bars lower it.")

            show = contrib.rename(columns={"feature": "Feature", "value": "Value", "contribution_pp": "Effect (points)"})
            st.dataframe(show.round(1), hide_index=True)
            st.caption("This estimate comes from a student project trained on partly synthetic data. "
                       "It is not a diagnosis. Talk to a doctor about any health concern.")

# ---- tab 2: performance ------------------------------------------------------------
with tab_perf:
    st.subheader("How well does the model work?")
    scr = meta["best_model_test_at_screening_threshold"]
    ci = meta["best_model_test_auc_ci95"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test ROC-AUC", f"{test['roc_auc']:.3f}", help=f"95% bootstrap interval {ci[0]:.3f} to {ci[1]:.3f}")
    m2.metric("Accuracy (cut-off 0.50)", f"{test['accuracy']:.2f}")
    m3.metric("Recall at screening cut-off", f"{scr['recall']:.2f}")
    m4.metric("Specificity at screening cut-off", f"{scr['specificity']:.2f}")

    comp = pd.read_csv(C.COMPARISON_PATH)
    keep = ["model", "cv_roc_auc", "test_roc_auc", "test_accuracy", "test_precision", "test_recall", "test_f1"]
    st.markdown("**Three models compared** (cross-validated AUC on the training data, the rest on the untouched test set)")
    st.dataframe(comp[keep].round(3), hide_index=True)

    by_src = meta["diagnostics"]["best_model_test_auc_by_source"]
    st.warning(
        f"**Read this before quoting the numbers.** On the 500 original rows the model scores an AUC of "
        f"about {by_src['original']['roc_auc']:.2f} (chance), on the synthetic rows about "
        f"{by_src['synthetic']['roc_auc']:.2f}. The headline score comes from relationships that were built into the "
        "generated data. It shows that the pipeline works, not that the model would work on real patients.")

    cols = st.columns(2)
    for col, (name, caption) in zip(cols * 2, [("roc_curves", "ROC curves"), ("confusion_screening", "Confusion matrix at the screening cut-off"),
                                              ("shap_bar", "Average SHAP importance"), ("shap_beeswarm", "SHAP summary")]):
        path = figure_path(name)
        if path:
            col.image(path, caption=caption)

# ---- tab 3: about ---------------------------------------------------------------------
with tab_about:
    st.subheader("Data")
    st.markdown(
        f"""
- **Size:** {meta['data']['n_rows']:,} patients with six features (age, BMI, blood pressure, glucose, cholesterol, smoking) and one label (disease / no disease).
- **Origin:** the first 500 rows (P0001-P0500) come from the project's provided file. The other 4,500 rows were **generated** for this project, using an *assumed* risk model in which older age, higher BMI, blood pressure, glucose, cholesterol and smoking raise the chance of disease.
- **Label:** one combined yes/no label. The data does not say which disease it stands for.
- **Balance:** the full file is exactly 50% disease and 50% no disease. Models also use class weights, and results are reported with recall, F1 and PR-AUC.
"""
    )
    st.subheader("Ethics and limits")
    st.markdown(
        """
- This tool must **not** be used for diagnosis or treatment decisions without supervision by qualified medical professionals.
- No real personal data is used. Patient ids are anonymous codes, and nothing typed into this page is stored.
- The model only sees six numbers. It knows nothing about symptoms, family history, medicines or lab tests, so it can miss patients and flag healthy ones.
- The screening threshold trades more false alarms for fewer missed cases. A real programme would set it together with doctors.
- Results come from one random split of partly synthetic data, with no external validation.
"""
    )
    st.subheader("How the estimate is explained")
    st.write("Each bar in the *Risk check* tab is a SHAP value in percentage points: how much that feature "
             "moved this patient's estimate away from the average reference patient.")
