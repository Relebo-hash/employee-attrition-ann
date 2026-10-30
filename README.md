# Employee Attrition Prediction Using an Artificial Neural Network

Mini project for **Neural Networks and Deep Learning** (MBA Data Science & AI, Chandigarh University)
**Student:** Relebohile Irene Lesitsi (25MDA10001)

## About
Extends Experiment 4 (ANN with SQLite data loading) into a full attrition-prediction study on the IBM HR Analytics dataset
(1,470 employees, 16.1% left). The baseline ANN over-fits; it is improved with dropout, L2 regularisation, early stopping,
class weights and a validation-selected decision threshold, and compared with Logistic Regression and Random Forest.

## Key results (test set, mean of 10 seeds)
| Model | ROC-AUC | F1 (leavers) |
|---|---|---|
| Baseline ANN (Exp. 4 setup) | 0.724 | 0.435 |
| Final ANN (tuned threshold) | 0.758 | 0.497 |
| Logistic Regression | 0.789 | 0.492 |

Logistic Regression ranks employees at least as well as the ANN; bootstrap confidence intervals show the difference is within sampling noise.
See the full report in `report/`.

## Files
- `Employee_Attrition_ANN_Mini_Project.ipynb` – complete executed notebook
- `data/` – dataset (IBM HR Analytics Employee Attrition; simulated data)
- `figures/`, `outputs/` – figures and result tables used in the report
- `models/` – saved final ANN, scaler and configuration
- `report/` – project report (Word)

## Run it
**Google Colab:** open the notebook from GitHub in Colab (Colab > File > Open notebook > GitHub tab) and choose *Runtime > Run all*.
The notebook downloads the dataset automatically if it is missing. The full run takes roughly 20-30 minutes on CPU
(mostly the hyper-parameter grid search).

**Locally:** `pip install -r requirements.txt`, then `jupyter notebook`.
