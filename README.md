# 📊 Telecom Churn Predictor

Welcome to the **Telecom Churn Predictor** repository! This project aims to identify which telecom customers are likely to churn, achieving an impressive accuracy of 94.60%. By leveraging engineered features from usage, billing, and support data, we implement various machine learning techniques to provide actionable insights for customer retention.

[![Download Releases](https://img.shields.io/badge/Download%20Releases-brightgreen)](https://github.com/gattsu001/Telecom-Churn-Predictor/releases)

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

Churn prediction is a critical task for telecom companies. Retaining customers is often more cost-effective than acquiring new ones. This project utilizes machine learning techniques to predict customer churn with high accuracy. 

The model combines several approaches, including decision trees and ensemble learning, to ensure robust predictions. By analyzing customer data, the model identifies patterns that lead to churn, helping businesses implement strategies to retain their customers.

## Key Features

- **High Accuracy**: Achieves 94.60% accuracy in churn prediction.
- **AUC Score**: Provides an AUC score of 0.8968, indicating strong model performance.
- **Precision and Recall**: Delivers a precision of 0.8675 and recall of 0.7423.
- **Feature Engineering**: Implements Sturges-based binning and one-hot encoding for effective feature extraction.
- **Stratified Sampling**: Uses an 80/20 train-test split to maintain the distribution of classes.
- **Ensemble Learning**: Employs a two-level ensemble pipeline with soft voting for improved predictions.

## Technologies Used

This project utilizes a range of technologies and libraries:

- **Python**: The primary programming language for model development.
- **Scikit-learn**: A machine learning library for building and evaluating models.
- **XGBoost**: An optimized gradient boosting library for efficient learning.
- **Joblib**: Used for model serialization and saving.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.

## Installation

To get started with the Telecom Churn Predictor, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/gattsu001/Telecom-Churn-Predictor.git
   cd Telecom-Churn-Predictor
   ```

2. **Install Required Packages**:

   Make sure you have Python installed. Then, install the necessary packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the Telecom Churn Predictor, follow these steps:

1. **Prepare Your Data**: Ensure your dataset is formatted correctly. The model expects specific features related to customer usage, billing, and support.

2. **Load the Model**: Use Joblib to load the pre-trained model.

   ```python
   from joblib import load

   model = load('model.joblib')
   ```

3. **Make Predictions**: Input your customer data into the model to predict churn.

   ```python
   predictions = model.predict(new_customer_data)
   ```

4. **Analyze Results**: Review the predictions to identify at-risk customers.

## Model Evaluation

Evaluating the model is crucial to ensure its effectiveness. The following metrics were used:

- **Accuracy**: Measures the proportion of true results among the total cases examined.
- **AUC (Area Under the Curve)**: Evaluates the model's ability to distinguish between classes.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: The ratio of correctly predicted positive observations to all actual positives.

## Results

The model demonstrated strong performance across various metrics:

- **Accuracy**: 94.60%
- **AUC**: 0.8968
- **Precision**: 0.8675
- **Recall**: 0.7423

These results indicate that the model effectively identifies customers likely to churn, providing valuable insights for retention strategies.

---

## SVM Contribution

This contribution adds a simple **Support Vector Machine (SVM)** baseline model for predicting telecom customer churn using the existing dataset `churn-in-telecom-dataset.csv`.

### Overview
The new SVM implementation provides a transparent and modular baseline that complements the existing ensemble-based approaches. It focuses on numeric features and offers an interpretable comparison point for evaluating the dataset’s separability and feature scaling effects.

### How to Run
From the repository root, execute:

```bash
python -m churn_svm_model.svm --csv churn-in-telecom-dataset.csv --label churn
```

### SVM Results
After running the above command, you will see a classification report like this:

               precision    recall  f1-score   support
   False            0.93      0.87      0.90       570
   True             0.46      0.64      0.54        97
   accuracy                             0.84       667
   macro avg        0.70      0.76      0.72       667
   weighted avg     0.87      0.84      0.85       667

### SVM Testing
A minimal test script `svm_test/test_svm.py` has been added to validate the SVM functionality.

It creates a small synthetic dataset to confirm that the function  
`train_and_evaluate()` runs successfully and produces a classification report.

Run the test with:
```bash
pytest -q
```

Expected output:

```bash
1 passed in 1.8s
```

## Contributing

We welcome contributions to improve the Telecom Churn Predictor. If you have suggestions or improvements, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Create a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, please reach out to:

- **GitHub**: [gattsu001](https://github.com/gattsu001)
- **Email**: gattsu001@example.com

For more information, visit our [Releases section](https://github.com/gattsu001/Telecom-Churn-Predictor/releases) to download the latest version of the model and documentation.

Thank you for your interest in the Telecom Churn Predictor! We hope this project helps you understand customer churn and develop effective retention strategies.