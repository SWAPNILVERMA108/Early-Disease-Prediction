# Early Disease Prediction Using Machine Learning on Health Data

HCLTech Career Shaper Programme · III Year Industry Project No. 29 · Digital-Data-AI / AIML

A complete machine learning project that estimates the chance of disease from six routine measurements (age, BMI, blood pressure, glucose, cholesterol, smoking) and explains each estimate with SHAP. It compares Logistic Regression, Random Forest and XGBoost, and includes a Streamlit app, three executed Jupyter notebooks, 34 automated tests, a report and an ethical declaration.

> **Disclaimer.** This is a student research prototype built on **partly synthetic data**. It must **not** be used for clinical diagnosis or medical decisions without supervision by qualified medical professionals.

## Quick start

You need Python 3.10 or newer. From this folder:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py        # start the app
jupyter notebook notebooks/               # open the notebooks
python -m pytest                          # run the 34 tests
 python -m src.train                      # optional: retrain everything (about 2 minutes)
```

The trained model is already included in `models/`, so the app and the notebooks work without retraining. If the saved model cannot be loaded on your computer (for example because your scikit-learn version differs), the code refits the selected model automatically in a few seconds.

## What is inside

| Folder / file | Contents |
| --- | --- |
| `data/dataset_5000.csv` | 5,000 patients: the 500 original rows (P0001-P0500, unchanged) plus 4,500 generated rows (P0501-P5000) |
| `data/dataset.csv` | The 500 original rows as supplied |
| `data/data_dictionary.csv` | All 8 columns described |
| `data/processed/` | The exact train (4,000) and test (1,000) rows used |
| `notebooks/01_EDA.ipynb` | Data checks and 6 visualisations |
| `notebooks/02_Model_Training.ipynb` | Tuning and comparing the three models, screening threshold, learning curve |
| `notebooks/03_Evaluation_and_SHAP.ipynb` | Confidence intervals, permutation importance, SHAP, subgroup and error analysis |
| `src/` | Reusable code: `config`, `data_utils`, `models`, `train`, `evaluate`, `explain`, `predict`, `viz`, `generate_synthetic_rows` |
| `app/streamlit_app.py` | The prototype: risk estimate, SHAP explanation, model performance, ethics page |
| `models/` | Saved models (`best_model.joblib` is the selected one) |
| `reports/Project_Report.docx` (+ `.pdf`) | The 19-page project report |
| `reports/metrics.json`, `model_comparison.csv`, `figures/` | All results and figures |
| `tests/` | 34 automated tests |
| `docs/Ethical_Compliance_Declaration.docx` | Declaration to fill in and sign |
| `docs/Demo_Script.md` | Script for the 1 to 2 minute demo video |

## Results at a glance

| Model | CV ROC-AUC | Test ROC-AUC | Accuracy | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| **Logistic Regression** (selected) | 0.816 | **0.822** | 0.749 | 0.750 | 0.749 |
| Random Forest | 0.806 | 0.816 | 0.743 | 0.748 | 0.744 |
| XGBoost | 0.810 | 0.815 | 0.734 | 0.734 | 0.734 |
| Majority-class baseline | - | 0.500 | 0.500 | 0.000 | - |

- The selected model catches about 85% of sick test patients at the screening cut-off of 0.39 (specificity 0.63).
- Glucose is the dominant feature in every method (permutation importance, SHAP, coefficients).

## Read this before quoting the numbers

1. **The data is partly synthetic.** The 4,500 extra rows were generated with an *assumed* risk model (older age, higher BMI, blood pressure, glucose, cholesterol and smoking raise the chance of disease).
2. **The 500 original rows carry no learnable signal.** Models trained only on them score a cross-validated AUC of about 0.45 to 0.50, which is chance. The final model scores about 0.50 on the original test rows and about 0.86 on the generated test rows. So the headline AUC shows that the pipeline works, not that the model would work on real patients.
3. **One combined label.** The brief names diabetes, heart disease and hypertension, but the file has a single yes/no label.
4. **Brief's reference link.** The UCI link in the brief (dataset 34) is the "Diabetes" dataset (AIM '94, timestamped insulin and glucose logs), not the Pima Indians data. Ask your mentor to confirm the reference.

## Deliverables from the brief

| Deliverable | Where it is | Status |
| --- | --- | --- |
| Working prototype | `app/`, `src/` | Done |
| Project report, minimum 8 pages | `reports/Project_Report.docx` | Done (19 pages). **Fill in** your name, roll number, college, mentor and date on the cover page (highlighted in yellow) |
| EDA notebook, 3+ visualisations | `notebooks/01_EDA.ipynb` | Done (6 visualisations) |
| Cleaned dataset | `data/dataset_5000.csv` | Done. Cleaning removed nothing because the file already passes all checks. Rename it to `dataset.csv` if the programme needs that exact file name |
| `data_dictionary.csv` | `data/data_dictionary.csv` | Done |
| Demo video, 1 to 2 minutes | `docs/Demo_Script.md` | **You need to record it** (script and timings are ready) |
| Signed Ethical Compliance Declaration | `docs/Ethical_Compliance_Declaration.docx` | **You need to fill in and sign it** |

## How the work is organised

1. **Validate and clean** (`src/data_utils.py`): wrong columns, missing values, duplicates and out-of-range values are checked. Bad rows are dropped, never changed.
2. **Split once**: 80% train / 20% test, stratified by label and by data source.
3. **Tune three models** with 5-fold cross-validation on the training rows only. Scaling lives inside the pipeline, so nothing leaks from validation or test rows.
4. **Select by cross-validation**, not by the test score. Choose the screening threshold from out-of-fold training predictions.
5. **Test once**, with bootstrap confidence intervals, subgroup checks and error analysis.
6. **Explain** with SHAP (globally, per model, and per patient in percentage points).
7. **Serve** the selected model in the Streamlit app, with input validation and the disclaimer.

All random choices use seed 42, so `python -m src.train` reproduces the reported numbers.

## Regenerating the extra rows

```bash
python src/generate_synthetic_rows.py data/dataset.csv data/dataset_5000.csv
```

The script is deterministic (seed 42). Running it again with a different seed gives different generated rows, and all results would change.

## Ethics in one paragraph

No real patients are involved and nothing typed into the app is stored. The data is described as partly synthetic everywhere, the original-versus-generated finding is reported openly, the screening threshold makes the trade-off between missed patients and false alarms visible, and performance is checked across age, smoking and BMI groups. Parts of the code and documents were prepared with the help of an AI assistant (Claude, by Anthropic), and this is stated in the report and the declaration.
