# Ensembles: Advanced Ensemble Learning for Imbalanced Datasets

This repository, **OrShulrufer/Ensembles**, provides a specialized implementation of ensemble learning techniques to address the challenges of highly imbalanced datasets. By leveraging dataset balancing, probabilistic calibration, and ensemble strategies, it ensures robust predictions for critical applications.

---

## Repository Structure

- **`__init__.py`**: Initializes the package, exposing the `EnsembleClassifier`.
- **`ensemble_classifier.py`**: Core implementation file containing the `EnsembleClassifier` class, equipped with methods for dataset balancing, classifier optimization, and calibrated voting mechanisms.
- **`test.py`**: Testing framework with synthetic dataset generation and performance evaluation, including visualization tools.
- **`requirements.txt`**: Contains all necessary dependencies for easy installation.
- **`README.md`**: Documentation outlining the project, structure, and usage.

---

## Features

- **Custom Voting Classifier**: Integrates multiple classifiers to achieve robust, reliable predictions.
- **Dataset Balancing**: Creates balanced subsets dynamically for training on imbalanced datasets.
- **Calibration**: Ensures probabilistic outputs are accurate using the sigmoid calibration method.
- **Synthetic Dataset Testing**: Provides tools to evaluate the methodology under controlled imbalanced scenarios.

---

## Prerequisites

- Python 3.8 or later
- Install dependencies using:
  ```bash
  pip install -r requirements.txt
Usage
Running the Ensemble Classifier
The ensemble_classifier.py file includes the EnsembleClassifier class, which can be imported into your project or directly tested using test.py.

Testing the Classifier
The test.py script demonstrates the classifier's performance:

Dataset Generation: Creates a synthetic dataset with 100,000 samples, 20 features, 15 informative features, and an imbalance ratio of 98% negatives to 2% positives.
Model Training: Trains the EnsembleClassifier on balanced subsets of the dataset.
Evaluation: Outputs metrics like accuracy, precision, recall, F1-score, and AUC.
Run the test with:

bash
Copy code
python test.py
Key Components
EnsembleClassifier: Includes methods such as:
_create_balanced_datasets: Dynamically creates balanced subsets.
_optimize_classifier: Optimizes individual classifiers for voting.
create_voting_classifier: Constructs the Voting Classifier.
create_calibrated_voting_classifier: Applies sigmoid calibration.
fit, predict_proba, predict: Core methods for training and inference.
Testing Tools:
generate_dataset: Synthetic dataset generation.
plot_balanced_comparison: Visualizes performance on balanced vs imbalanced data.
Results
The ensemble methodology addresses class imbalance effectively, ensuring calibrated probabilistic outputs and achieving high performance across various metrics. Visualization tools in test.py illustrate model performance.

Contributions
Contributions are welcome! Fork the repository, make improvements, and submit a pull request for review.

