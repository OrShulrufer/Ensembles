# Ensembles: Advanced Ensemble Learning for Imbalanced Datasets

This repository, **OrShulrufer/Ensembles**, implements advanced ensemble learning techniques to address the challenges posed by highly imbalanced datasets. It focuses on integrating multiple classifiers through a custom Voting Classifier, providing tools for balancing datasets, calibrating predictions, and rigorous evaluation.

## Repository Structure

- **`__init__.py`**: Initializes the package, importing the `EnsembleClassifier`.
- **`ensemble_classifier.py`**: Contains the implementation of the `EnsembleClassifier`, including dataset balancing, model training, and calibrated prediction methods.
- **`test.py`**: Provides a testing framework, generating synthetic datasets and evaluating the performance of the `EnsembleClassifier`.
- **`README.md`**: Documentation detailing the setup and usage of the repository.

## Features

- **Custom Voting Classifier**: Combines multiple machine learning models for enhanced predictions, ensuring flexibility and precision.
- **Dataset Balancing**: Automatically creates balanced subsets from imbalanced data for improved training.
- **Calibration**: Uses the sigmoid method to generate accurate probabilistic outputs.
- **Synthetic Dataset Testing**: Includes a dataset generator to evaluate the classifier's performance under controlled conditions.

## Getting Started

### Prerequisites
[Uploading requirements.txt…]()

Ensure Python 3.8 or later is installed. Required libraries include `pandas`, `numpy`, `scikit-learn`, and `joblib`. Install dependencies using the following command:

pip install -r requirements.txt
Installation and Usage
Clone the repository and navigate to its directory:

bash
Copy code
git clone https://github.com/OrShulrufer/Ensembles.git
cd Ensembles
The core implementation is available in ensemble_classifier.py. Import the EnsembleClassifier class into your project or use the provided test.py script to evaluate the methodology.

Running the Test
The test.py script demonstrates the classifier's capabilities on a synthetic dataset:

Generates a dataset with 100,000 samples, 20 features, 15 informative features, and 3 clusters per class, with an imbalance ratio of 98% negatives and 2% positives.
Trains the EnsembleClassifier on dynamically balanced subsets.
Evaluates the model using performance metrics such as accuracy, precision, recall, F1-score, and AUC.
Run the test with the following command:

bash
Copy code
python test.py
The script outputs a confusion matrix, AUC scores, and other metrics for analysis.

Results
The ensemble methodology effectively addresses the imbalance challenge, providing calibrated probabilistic predictions and achieving high performance across key metrics. Results from the test demonstrate the classifier's robustness in handling extreme class distributions.
