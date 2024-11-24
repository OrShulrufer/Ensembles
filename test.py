from ensemble_classifier import *


def generate_dataset():
    X, y = make_classification(
        n_samples=100_000,      # Total number of samples
        n_features=20,          # Total number of features
        n_informative=15,       # Number of informative features (more complexity)
        n_clusters_per_class=3, # Increased clusters per class for more complexity
        weights=[0.98, 0.02],   # Extreme class imbalance (2% positives)
        class_sep=0.5,          # Reduced class separability for harder predictions
        random_state=42         # For reproducibility
    )
    
    X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])]).astype(float)
    y = pd.Series(y, name="target", index=X.index) 

    print(f"Dataset created with {len(X)} samples and {X.shape[1]} features.")
    print(f"Class distribution: {y.value_counts(normalize=True).to_dict()}")

    return X, y



def plot_balanced_comparison(classifiers, X_test, y_test):
    metrics = {"Accuracy": [], "Balanced Accuracy": [], "Precision": [], "Recall": [], "F1-Score": [], "ROC AUC": [], "Overall Score": []}
    classifier_names = []
    weights = {"Accuracy": 0.2, "Balanced Accuracy": 0.25, "Precision": 0.15, "Recall": 0.15, "F1-Score": 0.2, "ROC AUC": 0.05}
    for clf_name, clf in classifiers:
        print('\n', '-----'*20, f'\nResults for {clf_name}\n', '-----'*20)
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba[:,1]) 
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        overall_score = (
            weights["Accuracy"] * accuracy + 
            weights["Balanced Accuracy"] * balanced_accuracy + 
            weights["Precision"] * precision + 
            weights["Recall"] * recall + 
            weights["F1-Score"] * f1 + 
            weights["ROC AUC"] * roc_auc
        )
        print(f"Overall Score: {overall_score:.4f}")
        classifier_names.append(clf_name)
        metrics["Accuracy"].append(accuracy)
        metrics["Balanced Accuracy"].append(balanced_accuracy)
        metrics["Precision"].append(precision)
        metrics["Recall"].append(recall)
        metrics["F1-Score"].append(f1)
        metrics["ROC AUC"].append(roc_auc)
        metrics["Overall Score"].append(overall_score)
    print(f"Ensemble best threshold: {classifiers[-1][1].best_threshold_:.4f}")
    metric_names = list(metrics.keys())
    x = np.arange(len(metric_names))
    width = 1 / (len(classifier_names) + 1) 
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(classifier_names)))
    for idx, clf_name in enumerate(classifier_names):
        offset = (idx - len(classifier_names) / 2) * width + width / 2
        ax.bar(x + offset, [metrics[metric][idx] for metric in metric_names], width, label=clf_name, color=colors[idx], alpha=0.85 if clf_name == "Ensemble" else 0.6, edgecolor='black' if clf_name == "Ensemble" else 'none', linewidth=1.5 if clf_name == "Ensemble" else 0)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Comparison of Classifier Metrics (Including Balanced Accuracy and Overall Score)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    legend = ax.legend(title="Classifiers", fontsize=10, loc="best", frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.3)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    X, y = generate_dataset()

    scaler = RobustScaler(quantile_range=(1.0, 99.0),  unit_variance=True)

    X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

    ensemble_clf = EnsembleClassifier()

    ensemble_clf.fit(X_train, y_train, ["RandomForestClassifier", "XGBClassifier", "ExtraTreesClassifier"])

    classifiers = []
    files = sorted(os.listdir(ensemble_clf.storage_folder_path), key=lambda x: int(x.split('_')[1].split('.')[0]) )

    for filename in files:
        if filename.endswith('.joblib'):
            model_path = os.path.join(ensemble_clf.storage_folder_path, filename)
            model_name = filename.rsplit('.', 1)[0]
            model = joblib.load(model_path)
            if not any(isinstance(existing_model[1], type(model)) for existing_model in classifiers):
                classifiers.append((model.__class__.__name__, model))

    # Unbalanced classifiers
    rf_unbalanced = RandomForestClassifier(random_state=42)
    xgb_unbalanced = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    et_unbalanced = ExtraTreesClassifier(random_state=42)
    rf_unbalanced.fit(X_train, y_train)
    xgb_unbalanced.fit(X_train, y_train)
    et_unbalanced.fit(X_train, y_train)
    classifiers.append(("RandomForestClassifier (Unbalanced)", rf_unbalanced))
    classifiers.append(("XGBClassifier (Unbalanced)", xgb_unbalanced))
    classifiers.append(("ExtraTreesClassifier (Unbalanced)", et_unbalanced))

    # Balanced classifiers
    rf_balanced = RandomForestClassifier(class_weight='balanced', random_state=42)
    xgb_balanced = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), random_state=42)
    et_balanced = ExtraTreesClassifier(class_weight='balanced', random_state=42)
    rf_balanced.fit(X_train, y_train)
    xgb_balanced.fit(X_train, y_train)
    et_balanced.fit(X_train, y_train)
    classifiers.append(("RandomForestClassifier (Balanced)", rf_balanced))
    classifiers.append(("XGBClassifier (Balanced)", xgb_balanced))
    classifiers.append(("ExtraTreesClassifier (Balanced)", et_balanced))


    classifiers.append(("Ensemble", ensemble_clf))

    plot_balanced_comparison(classifiers, X_test, y_test)