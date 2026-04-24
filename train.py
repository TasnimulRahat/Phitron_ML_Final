import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
     precision_score, 
     recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

df = pd.read_csv("Social_Network_Ads.csv")
df=df.drop("User ID", axis=1)
df=df.drop_duplicates()
X = df.drop("Purchased", axis=1)
y = df["Purchased"]
numerical_features = X.select_dtypes(include=[np.number]).columns
catagorical_features = X.select_dtypes(include=["object", "category"]).columns
for col in numerical_features:
    q1 = X[col].quantile(0.25)
    q3 = X[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    X[col] = np.where(X[col] < lower_bound, lower_bound, X[col])
    X[col] = np.where(X[col] > upper_bound, upper_bound, X[col])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, numerical_features),
    ("cat", cat_transformer, catagorical_features)
])
model = RandomForestClassifier(random_state=42,max_depth=None,n_estimators=100
  ,min_samples_split=5,n_jobs=-1)
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])
pipeline.fit(X_train, y_train)

y_pred_test = pipeline.predict(X_test)

print(classification_report(y_test, y_pred_test))
#pickle
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)