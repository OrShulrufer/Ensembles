
import os
import joblib
from matplotlib import pyplot as plt
import sklearn
import random
import pandas as pd
import numpy as np
import logging
from sklearn.calibration import CalibratedClassifierCV, LabelEncoder
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, auc, average_precision_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, make_scorer, precision_recall_curve, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
logging.basicConfig(filename='Ensembles/logs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss



class EnsembleClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, max_datasets=100, min_precision=None, min_recall=None):

        # Path for storing trained model files
        self.storage_folder_path = 'Ensembles/models'
        # Directory creation for storing model files, if not already existing
        os.makedirs(self.storage_folder_path, exist_ok=True)
    
        # initial and default parametrisations 
        self.models_ = []
        self.voting_clf_ = None
        self.cal_voting_clf_ = None
        self.best_threshold_ = None
        self.balanced_datasets_ = None
        self.max_datasets_ = max_datasets
        self.random_seed_ = 42
        self.min_precision = min_precision
        self.min_recall = min_recall

        # Hyperparameter configurations for different classifiers
        self.clf_dictionaries_ = {
                                "XGBClassifier": {
                                    "objective": "binary:logistic",
                                    "use_label_encoder": None,
                                    "base_score": None,
                                    "booster": "gbtree",
                                    "callbacks": None,
                                    "colsample_bylevel": 0.6,
                                    "colsample_bynode": 0.6,
                                    "colsample_bytree": 0.6,
                                    "early_stopping_rounds": None,
                                    "enable_categorical": False,
                                    "eval_metric": "logloss",
                                    "feature_types": None,
                                    "gamma": 0,
                                    "gpu_id": None,
                                    "grow_policy": 'depthwise',      
                                    "importance_type": "gain",
                                    "interaction_constraints": None,
                                    "learning_rate": 0.3,             
                                    "max_bin": None,
                                    "max_cat_threshold": None,
                                    "max_cat_to_onehot": None,
                                    "max_delta_step": None,
                                    "max_depth": 6,                  
                                    "max_leaves": None,
                                    "min_child_weight": 1,           
                                    "monotone_constraints": None,  
                                    "n_estimators": 100,
                                    "n_jobs": 1,
                                    "num_parallel_tree": 5,
                                    "random_state": 42,
                                    "reg_alpha": 0,
                                    "reg_lambda": 0,
                                    "sampling_method": 'uniform',  
                                    "scale_pos_weight": 1,
                                    "subsample": 1.0,               
                                    "tree_method": 'hist',
                                    "validate_parameters": False,
                                    "verbosity": 0
                                },
                                "ExtraTreesClassifier": {
                                    "bootstrap": True,
                                    "ccp_alpha": 0.,
                                    "class_weight": None,
                                    "criterion": "gini",
                                    "max_depth": 6,
                                    "max_features": 'sqrt',
                                    "max_leaf_nodes": None,
                                    "max_samples": 1.0,               
                                    "min_impurity_decrease": 0.0,
                                    "min_samples_leaf": 6,
                                    "min_samples_split": 8,
                                    "min_weight_fraction_leaf": 0.0,
                                    "n_estimators": 100,
                                    "n_jobs": 1,
                                    "oob_score": True,
                                    "random_state": 42,
                                    "verbose": 0,
                                    "warm_start": False
                                },
                                "RandomForestClassifier": {
                                    "bootstrap": True,
                                    "ccp_alpha": 0.0,
                                    "class_weight": None,
                                    "criterion": "gini",
                                    "max_depth": 6,
                                    "max_features": 'sqrt',
                                    "max_leaf_nodes": None,
                                    "max_samples": 1.0,             
                                    "min_impurity_decrease": 0.0,
                                    "min_samples_leaf": 6,
                                    "min_samples_split": 8,
                                    "min_weight_fraction_leaf": 0.0,
                                    "n_estimators": 100,
                                    "n_jobs": 1,
                                    "oob_score": True,
                                    "random_state": 42,
                                    "verbose": 0,
                                    "warm_start": False
                                }
                            }
        self.param_grids_ = {
                            'RandomForestClassifier': {
                                'oob_score':[True, False],
                                'max_features': ['sqrt', 'log2'],
                                'min_samples_leaf': [1, 3, 6, 9, 12, 16],
                                'min_samples_split': [2, 4, 8, 12, 16, 20],
                                'max_depth': [6, 9, 12],
                                'max_samples': [0.6, 0.8, 1.0],
                                'criterion': ['gini', 'log_loss'] 
                                },
                            'XGBClassifier': {
                                'learning_rate': [0.3, 0.03],
                                'colsample_bytree': [0.6, 0.8, 1.0],
                                'colsample_bylevel': [0.6, 0.8, 1.0],
                                'colsample_bynode': [0.6, 0.8, 1.0],
                                'max_depth': [6, 9, 12],
                                'subsample': [0.6, 0.8, 1.0],
                                'monotone_constraints': [None, (1,-1)],
                                'grow_policy': ['depthwise', 'lossguide']
                                },
                            'ExtraTreesClassifier': {
                                'oob_score':[True, False],
                                'max_features': ['sqrt', 'log2'],
                                'min_samples_leaf': [1,3,6, 9, 12, 16],
                                'min_samples_split': [2, 4, 8, 12, 16, 20],
                                'max_depth': [6, 9, 12],
                                'max_samples': [0.6, 0.8, 1.0],
                                'criterion': ['gini', 'log_loss']                             
                                }
                            }
        
        # Factory of initial classifiers classifiers 
        self.classifiers_map = {
            "RandomForestClassifier": sklearn.clone(RandomForestClassifier(**self.clf_dictionaries_['RandomForestClassifier'])),
            "ExtraTreesClassifier": sklearn.clone(ExtraTreesClassifier(**self.clf_dictionaries_['ExtraTreesClassifier'])),
            "XGBClassifier": sklearn.clone(XGBClassifier(**self.clf_dictionaries_['XGBClassifier']))
            }
        
        # Definition of a custom scorer for evaluating and guiding optimisation of classifiers
        self.scorers_ = {
            'custom_scorer': make_scorer(lambda y_true, y_prob: 
                                         0.3 * roc_auc_score(y_true, y_prob) + 
                                         0.4 * average_precision_score(y_true, y_prob) + 
                                         0.3 * auc(*precision_recall_curve(y_true, y_prob)[1::-1]) 
                                         ,greater_is_better=True
                                         ,response_method='predict_proba')
            }


    def _create_balanced_datasets(self):

        # Pool of negative samples extracted from the training data
        negative_pool = self.samples_[self.targets_ == 0].copy()
        negative_pool_target = self.targets_[self.targets_ == 0].copy()

        # Positive samples extracted from the training data
        pos_df = self.samples_[self.targets_ == 1].copy()
        pos_target = self.targets_[self.targets_ == 1].copy()

        # Length of positive samples and the number of datasets required
        neg_length = len(pos_df)
        num_datasets = (len(self.samples_) // len(pos_df)) - 1

        # Indexes for sampling negative data to create balanced datasets
        negative_pool_index = negative_pool.sample(n=num_datasets * neg_length, random_state=self.random_seed_).index
         
        # Creation of balanced datasets by combining positives and negatives, 
        # each iteration removing negatives that were sampled from negatives pool.
        datasets = []
        for i in range(num_datasets):

            negatives = negative_pool.loc[negative_pool_index[i * neg_length : (i + 1) * neg_length]]

            negatives_targets = negative_pool_target.loc[negative_pool_index[i * neg_length : (i + 1) * neg_length]]

            X_balanced = pd.concat([pos_df.copy(), negatives.copy()], axis=0, ignore_index=True)
            y_balanced = pd.concat([pos_target.copy(), negatives_targets.copy()], axis=0, ignore_index=True)

            X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=self.random_seed_)

            datasets.append((X_balanced, y_balanced))

        self.balanced_datasets_ = datasets

        return datasets
    

    def _optimize_classifier(self, clf, X, y):

        # Get predefined hyper-parameters grid for optimisation class to match.
        param_grid = self.param_grids_[clf.__class__.__name__]

         # Cross-validation strategy with stratified sampling
        cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=self.random_seed_)

        print("\n","-----"*20,f'\nStarting Optimization of {clf}\n', "-----"*20)

        # Start of iterative grid search to refine hyperparameters
        need_improve = True
        while need_improve:
            need_improve = False

            for param_name, param_values in param_grid.items():

                temp_param_grid = {param_name: param_values}

                grid_search = GridSearchCV(
                    estimator=clf,
                    param_grid=temp_param_grid,
                    cv=cv,
                    scoring=self.scorers_,
                    refit="custom_scorer",
                    n_jobs=-3,
                    verbose=0,
                    error_score="raise"
                )
                grid_search.fit(X, y)

                old_params = clf.get_params()
                new_params = grid_search.best_params_
                clf = grid_search.best_estimator_

                # If one of the parameters changed in this adjustment cycle, 
                # than another adjustment cycle needed to reach the local maximum optimisation point
                if old_params[param_name] != new_params[param_name]:

                    print(f"Switching {param_name} from {old_params[param_name]} to {new_params[param_name]}: {grid_search.best_score_}")

                    need_improve = True

            if need_improve:
                print("\n","-----"*20,"\nAnother loop needed\n", "-----"*20)
            else:
                print(f"Best parameters for {clf.__class__.__name__} are {clf.get_params()}")

        return clf


    def create_voting_classifier(self):
        models = []
        # Loading models from storage ensures consistency with previously trained and saved configurations
        # Models were saved to handle scalability, allowing reuse without retraining across sessions or large datasets
        files = sorted(os.listdir(self.storage_folder_path), key=lambda x: int(x.split('_')[1].split('.')[0]) )
        for filename in files:
            if filename.endswith('.joblib'):
                model_path = os.path.join(self.storage_folder_path, filename)
                model_name = filename.rsplit('.', 1)[0]
                model = joblib.load(model_path)
                if isinstance(model, (XGBClassifier, RandomForestClassifier, ExtraTreesClassifier)):
                    models.append((model_name, model))

        # Configures the voting classifier with pre-trained models and label encodings, 
        # ensuring compatibility for prediction without retraining
        self.voting_clf_ = VotingClassifier(estimators=models, voting="soft")
        self.voting_clf_.estimators_ = [m[1] for m in models]
        self.voting_clf_.le_ = LabelEncoder().fit(self.targets_)
        self.voting_clf_.classes_ = self.voting_clf_.le_.classes_
        return self.voting_clf_
    

    def create_calibrated_voting_classifier(self):

        if self.voting_clf_ is None:
            self.create_voting_classifier()

        self.cal_voting_clf_ = CalibratedClassifierCV(self.voting_clf_, cv='prefit', method="sigmoid", ensemble=True, n_jobs=-2)

        X_val = pd.concat([self.cal_samples_, self.samples_])
        y_val = pd.concat([self.cal_targets_, self.targets_])
        # X_val, y_val = self.cal_samples_, self.cal_targets_

        X_val, y_val = NearMiss(version=3).fit_resample(X_val, y_val)

        self.cal_voting_clf_.fit(X_val, y_val)

        # Calculation of the best threshold for binary decision-making based on max F1 score possible
        y_prob_best_threshold = self.cal_voting_clf_.predict_proba(self.bt_samples_)[:,1]
        precisions, recalls, thresholds = precision_recall_curve(self.bt_targets_ , y_prob_best_threshold)

        # If needed some minimum precision level in predictions
        if self.min_precision is not None:
            mask = precisions[:-1] >= self.min_precision 
            precisions = precisions[mask]
            recalls = recalls[mask]
            thresholds = thresholds[mask]
        # If needed some minimum recall level in predictions
        elif self.min_recall is not None:
            mask = recalls[:-1] >= self.min_recall
            precisions = precisions[mask]
            recalls = recalls[mask]
            thresholds = thresholds[mask]


        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        self.best_threshold_ = thresholds[np.argmax(f1_scores)]
        

        return self.cal_voting_clf_
    

    def fit(self, X, y, classifiers):

        # First of two splits forgetting training, calibration, and best threshold datasets with stratification 
        self.samples_, X_test, self.targets_, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=self.random_seed_)

        # Second of two splits forgetting training, calibration, and best threshold datasets with stratification 
        self.cal_samples_, self.bt_samples_, self.cal_targets_, self.bt_targets_ = train_test_split(X_test, y_test, test_size=0.2, stratify=y_test, random_state=self.random_seed_)

        print(f"Samples: Positives={self.targets_.sum()}, Negatives={len(self.targets_) - self.targets_.sum()}, Calibration Samples: Positives={self.cal_targets_.sum()}, Negatives={len(self.cal_targets_) - self.cal_targets_.sum()}, Best Threshold Samples: Positives={self.bt_targets_.sum()}, Negatives={len(self.bt_targets_) - self.bt_targets_.sum()}")

        # Creation and capping of balanced datasets
        datasets = self._create_balanced_datasets()

        # Limit Ensemble size to particle size
        if len(datasets) > self.max_datasets_:
            datasets = datasets[ :self.max_datasets_]

        # Training, optimization and storage of classifiers types for each balanced dataset
        for clf_num , (X, y) in enumerate(datasets):

            for clf_class_name in classifiers:

                initial_clf = self.classifiers_map[clf_class_name]

                file_name = os.path.join(self.storage_folder_path, f"{clf_class_name}_{clf_num}.joblib")

                if not os.path.exists(file_name):
                    best_trained_model = self._optimize_classifier(initial_clf, X, y)

                    self.models_.append((file_name, best_trained_model))

                    joblib.dump(best_trained_model, file_name)

                print(f'Finished Optimizing "{clf_class_name}_{clf_num}" Model')

        self.create_calibrated_voting_classifier()
    

    def predict_proba(self, X):
        if self.cal_voting_clf_ is None:
            if sum(1 for file_name in os.listdir(self.storage_folder_path) if file_name.endswith('.joblib')) >= 3:
                self.create_calibrated_voting_classifier()
            else: 
                return None
        return self.cal_voting_clf_.predict_proba(X)
    

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.best_threshold_).astype(int)




