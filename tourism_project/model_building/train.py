# for data manipulation
import pandas as pd
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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score


api = HfApi()

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



param_grid = {
    'xgbclassifier__n_estimators': [100, 200],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
    'xgbclassifier__min_child_weight': [1, 3],
    'xgbclassifier__subsample': [0.8, 1],
    'xgbclassifier__colsample_bytree': [0.8, 1],
    'xgbclassifier__gamma': [0, 0.1, 0.2],
    'xgbclassifier__reg_alpha': [0, 0.1, 1],
    'xgbclassifier__reg_lambda': [1, 1.5, 2]
}

random_search = RandomizedSearchCV(
    estimator=model_pipeline,
    param_distributions=param_grid,
    n_iter=100,               
    scoring=make_scorer(f1_score, pos_label=1),
    cv=3,
    random_state=42,
    verbose=2,
    n_jobs=-1
)

random_search.fit(Xtrain, ytrain)
best_model = random_search.best_estimator_

print("Best Params:\n", random_search.best_params_)

# Predict on training set
y_pred_train = best_model.predict(Xtrain)

# Predict on test set
y_pred_test = best_model.predict(Xtest)

# Evaluation
print("\nTraining Classification Report:")
print(classification_report(ytrain, y_pred_train))

print("\nTest Classification Report:")
print(classification_report(ytest, y_pred_test))


# Get transformed feature names after fitting
ohe_feature_names = (
    best_model.named_steps['columntransformer']
    .named_transformers_['onehotencoder']
    .get_feature_names_out(categorical_features)
)

final_feature_names = numeric_features + list(ohe_feature_names)


print(final_feature_names)

joblib.dump(final_feature_names, "model_columns.joblib")





# Save best model
joblib.dump(best_model, "best_tourism_package_purchase_model_v1.joblib")

# Upload to Hugging Face
repo_id = "bala-ai/tourism_package_purchase_model"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

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

api.upload_file(
    path_or_fileobj="model_columns.joblib",
    path_in_repo="model_columns.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
