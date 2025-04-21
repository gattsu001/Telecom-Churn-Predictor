# Telecom Churn Predictor

**Churn classification model for telecom customer datasets.  
94.60% Accuracy | 0.8968 AUC | 0.8675 Precision | 0.7423 Recall**

Predicts which customers are likely to leave using a stacked ensemble of four classifiers.  
Built with real telecom data, trained with stratified validation, and fully reproducible.  
This repository includes a complete pipeline: feature engineering, model stacking, and evaluation.

---

## What It Does

This model predicts customer churn for telecom operators.  
It learns from customer usage patterns, billing behavior, service plans, and support interactions.  
Given raw input data, it outputs a churn probability between 0 and 1 for each customer.

The output helps retention teams target at-risk customers before they leave.

---

## Evaluation Metrics 

| Metric    | Value   | Description |
|-----------|---------|-------------|
| Accuracy  | 94.60%  | Overall correct predictions |
| AUC       | 0.8968  | Separation between churners and non-churners |
| Precision | 0.8675  | % of predicted churners that actually churned |
| Recall    | 0.7423  | % of actual churners correctly identified |

Evaluation done on 50/50 stratified train/test split.  
Base learners trained using k-fold CV. Meta-learner trained on out-of-fold predictions.

---

### Model Architecture: Two-Level Stacked Ensemble

The model is structured as a **two-tiered ensemble**, where each layer plays a distinct role in prediction.

#### **Level 1: Base Learners**

At the first level, four different classifiers are trained independently on the same input features:

- `XGBoostClassifier`
- `LogisticRegression`
- `DecisionTreeClassifier`
- `GaussianNB`

Each model learns its own logic for predicting churn based on customer attributes (usage, billing, service plans, etc.).  
When given a new customer record, each model outputs a **single probability value**—the likelihood that the customer will churn.

Think of each base model as casting a probabilistic vote:  
> “Given this input, I estimate the churn risk is 18%.”  
> “I estimate 34%.”  
> “I say 12%.”  
> “I predict 22%.”

This results in four probabilities per customer, one from each base learner.

#### **Level 2: Meta-Learner**

These four churn probabilities become the input to a **Logistic Regression meta-learner**.

The meta-learner doesn't see the original customer data. Instead, it learns how to **weight and combine** the base models' outputs based on historical performance:

- If XGBoost is usually right in high-risk cases, it gets more influence there.
- If Naive Bayes tends to overpredict churn, the meta-learner learns to downweight it.

The output of the meta-learner is the **final churn probability**, integrating the strengths of all four base models while reducing individual weaknesses.

This architecture improves generalization and avoids over-reliance on any single model’s assumptions or biases.


This stacked ensemble method leverages the strengths of each base model, aiming to improve overall predictive performance.

## Reproducibility > Claims

Publishing performance numbers in a PDF means nothing without **verifiable code**, **data**, and **methodology**.  
Claims like “99.28% accuracy” are scientifically meaningless without reproducibility.

In churn or NBO—where data is tabular, class distributions are imbalanced, and evaluation choices impact every number—**reproducibility is everything**.

GitHub models provide something better than publication in an academic journals: **proof**.

If you can inspect the code, run the pipeline, and trace the model's decisions on real data—  
you’ve got a **testable asset**, not a claim.


## Design Snapshot

| Component           | My Model                                             |
|---------------------|------------------------------------------------------|
| Bucketing Strategy  | Fixed 5-bin discretization for all numeric fields   |
| Ensemble Structure  | Single-stage `StackingClassifier` with all 4 models |
| Train/Test Split    | 50/50                                                |
| Feature Selection   | All available features, fully encoded               |
| Voting Mechanism    | Logistic Regression meta-learner                    |

---

## Conclusion

The final system is a layered, structured ensemble with strong performance and high transparency. Logistic regression as a meta-learner effectively balances outputs from diverse base classifiers, while quantile-based bucketing and complete categorical encoding ensure that the full information space is available during training.

A 94.6% accuracy and 0.8968 AUC make this implementation a strong benchmark for practical churn prediction. The modular architecture, clean feature processing, and documented evaluation steps support easy replication and extension—whether for production deployment or integration with retention strategy tools.

---

├── churn_model/
│   ├── predict.py
│   ├── train.py
│   ├── artifacts/
│   │   ├── xgb_model.joblib
│   │   ├── lr_model.joblib
│   │   ├── dt_model.joblib
│   │   ├── nb_model.joblib
│   │   └── preprocessor.joblib


---
## How to Run

```bash
git clone https://github.com/yourusername/telecom-churn-predictor.git
cd telecom-churn-predictor
pip install -r requirements.txt
python train.py
