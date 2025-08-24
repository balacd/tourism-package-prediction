# for data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score
import mlflow
from sklearn.metrics import precision_recall_curve, f1_score
import mlflow.sklearn
import sys
import shap
import matplotlib.pyplot as plt
import sklearn

hf_token = os.getenv("HF_TOKEN")
print("HF_TOKEN:", hf_token)

if not hf_token:
    raise ValueError("HF_TOKEN not found.")


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-experiment")


Xtrain_path = "hf://datasets/bala-ai/tourism-package-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/bala-ai/tourism-package-prediction/Xtest.csv"
ytrain_path = "hf://datasets/bala-ai/tourism-package-prediction/ytrain.csv"
ytest_path = "hf://datasets/bala-ai/tourism-package-prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


numeric_features = Xtrain.select_dtypes(include='number').columns.tolist()
categorical_features = Xtrain.select_dtypes(include='object').columns.tolist()


# Class weight to handle imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

# Preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)


# Define XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)


# Create pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)


# ---------------- Hyperparameter Grid ----------------
param_grid = {
    'xgbclassifier__n_estimators': [100, 200, 300],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__min_child_weight': [1, 3],
    'xgbclassifier__subsample': [0.8, 1],
    'xgbclassifier__colsample_bytree': [0.8, 1],
    'xgbclassifier__gamma': [0, 0.1],
    'xgbclassifier__reg_alpha': [0, 0.1],
    'xgbclassifier__reg_lambda': [1, 2]
}

# Start MLflow run
with mlflow.start_run():

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=3,
        scoring=make_scorer(f1_score, pos_label=1),
        n_jobs=-1,
        verbose=2
    )




    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations and their mean test scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_score", mean_score)
            mlflow.log_metric("std_test_score", std_score)

#     pd.DataFrame(results).to_csv("cv_results.csv", index=False)
#     mlflow.log_artifact("cv_results.csv")


    # Log best parameters separately in main run
    mlflow.log_params(grid_search.best_params_)


    best_model = grid_search.best_estimator_

    print("Best Params:\n", grid_search.best_params_)


# calculate the threshold value
    probs = best_model.predict_proba(Xtest)[:, 1]
    prec, rec, thresh = precision_recall_curve(ytest, probs)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-10)

    best_idx = np.argmax(f1_scores)
    classification_threshold = thresh[best_idx] if best_idx < len(thresh) else 0.5

    print("Best Threshold for F1:", classification_threshold)
    mlflow.log_param("classification_threshold", float(classification_threshold))




#     classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    print("\nTraining Classification Report:", train_report)

    test_report = classification_report(ytest, y_pred_test, output_dict=True)
    print("\nTest Classification Report:", test_report)


    # Log the metrics for the best model
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
    })



    # Save best model
    model_path = "best_tourism_package_purchase_model_v1.joblib"

    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name="tourism_package_model")

    mlflow.log_param("python_version", sys.version)
    mlflow.log_param("sklearn_version", sklearn.__version__)
    mlflow.log_param("xgboost_version", xgb.__version__)




    print(f"Model saved as artifact at: {model_path}")



    # ---------------------------
    # SHAP Explainability
    # ---------------------------


    try:
        # Extract fitted XGB model from pipeline
        fitted_xgb = best_model.named_steps['xgbclassifier']

        # Transform the train set using the same preprocessor
        transformer = best_model.named_steps['columntransformer']
        transformed_Xtrain = transformer.transform(Xtrain)

        # Get feature names after preprocessing
        feature_names = []
        if hasattr(transformer, "get_feature_names_out"):
            feature_names = transformer.get_feature_names_out()
            print("feature_names:",feature_names)

        else:
            # fallback: use numeric+categorical if method unavailable
            feature_names = numeric_features + categorical_features
            print("feature_names_2:",feature_names)

        # Initialize SHAP TreeExplainer
        explainer = shap.TreeExplainer(fitted_xgb)
        shap_values = explainer.shap_values(transformed_Xtrain)

        # ---- Save SHAP values as CSV ----
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.to_csv("shap_values.csv", index=False)
        mlflow.log_artifact("shap_values.csv", artifact_path="explainability")

        # ---- SHAP Summary Plot ----
        shap.summary_plot(shap_values, transformed_Xtrain, feature_names=feature_names, show=False)
        shap_summary_file = "shap_summary.png"
        plt.savefig(shap_summary_file, dpi=300, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(shap_summary_file, artifact_path="explainability")

        # ---- SHAP Bar Plot ----
        shap.summary_plot(shap_values, transformed_Xtrain, feature_names=feature_names, plot_type="bar", show=False)
        shap_bar_file = "shap_feature_importance.png"
        plt.savefig(shap_bar_file, dpi=300, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(shap_bar_file, artifact_path="explainability")

        print("SHAP plots and values logged to MLflow.")


    except Exception as e:
        print("Failed to generate SHAP outputs:", e)




    # Upload to Hugging Face
    repo_id = "bala-ai/tourism_package_purchase_model"
    repo_type = "model"

    api = HfApi(token=hf_token)

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model Space '{repo_id}' created.")

    # create_repo("best_tourism_package_purchase_model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_tourism_package_purchase_model_v1.joblib",
        path_in_repo="best_tourism_package_purchase_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
