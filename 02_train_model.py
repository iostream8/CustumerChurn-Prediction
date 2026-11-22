"""Step 2 - Clean data, train 3 models, pick the best, check business value, save everything.
Runs on your laptop AND inside Google Cloud Vertex AI (same file)."""
import argparse, json, os, joblib, numpy as np, pandas as pd, sklearn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/customer_churn.csv")     # can also be gs://bucket/file.csv
ap.add_argument("--out", default="outputs")
args = ap.parse_args()

# Business assumptions (change these to match your company)
OFFER_COST, SAVE_RATE, CUSTOMER_VALUE = 50, 0.30, 600   # $ per offer, % who stay after offer, $ value of a saved customer

NUM = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "support_calls"]
CAT = ["gender", "Partner", "Dependents", "PhoneService", "InternetService", "OnlineSecurity",
       "TechSupport", "Contract", "PaperlessBilling", "PaymentMethod"]

# 1) Load + split
df = pd.read_csv(args.data)
X, y = df[NUM + CAT], (df["Churn"] == "Yes").astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2) Preprocessing: fill gaps, scale numbers, turn words into 0/1 columns
prep = ColumnTransformer([
    ("num", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)}

# 3) Compare models fairly with 5-fold cross-validation (only on training data)
scores = {}
for name, m in models.items():
    scores[name] = cross_val_score(Pipeline([("prep", prep), ("model", m)]), X_tr, y_tr, cv=5, scoring="roc_auc").mean()
    print(f"{name:20s} CV ROC-AUC = {scores[name]:.3f}")
best_name = max(scores, key=scores.get)
best = Pipeline([("prep", prep), ("model", models[best_name])]).fit(X_tr, y_tr)
print("Best model:", best_name)

# 4) Final exam on data the model never saw
proba = best.predict_proba(X_te)[:, 1]

def profit(t):
    pred = proba >= t
    tp = (pred & (y_te.values == 1)).sum()
    return tp * SAVE_RATE * CUSTOMER_VALUE - pred.sum() * OFFER_COST

thresholds = np.arange(0.05, 0.95, 0.01)
best_t = float(thresholds[np.argmax([profit(t) for t in thresholds])])

def report(t):
    p = (proba >= t).astype(int)
    return {"threshold": round(float(t), 2), "accuracy": accuracy_score(y_te, p), "precision": precision_score(y_te, p),
            "recall": recall_score(y_te, p), "f1": f1_score(y_te, p), "profit_usd": float(profit(t))}

metrics = {"best_model": best_name, "cv_auc": scores, "test_roc_auc": roc_auc_score(y_te, proba),
           "at_default_0.5": report(0.5), "at_profit_threshold": report(best_t),
           "chosen_threshold": round(best_t, 2), "profit_if_we_contact_nobody": 0.0,
           "profit_if_we_contact_everybody": float(profit(0.0)),
           "test_customers": int(len(y_te)), "test_churners": int(y_te.sum()), "sklearn_version": sklearn.__version__}
print(json.dumps(metrics, indent=2, default=float))

# 5) Charts
os.makedirs(args.out, exist_ok=True)
RocCurveDisplay.from_predictions(y_te, proba); plt.title("ROC curve"); plt.savefig(f"{args.out}/roc_curve.png", dpi=120); plt.close()
ConfusionMatrixDisplay.from_predictions(y_te, (proba >= best_t).astype(int), display_labels=["Stayed", "Churned"])
plt.title(f"Confusion matrix (threshold {best_t:.2f})"); plt.savefig(f"{args.out}/confusion_matrix.png", dpi=120); plt.close()
imp = permutation_importance(best, X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=42)
order = np.argsort(imp.importances_mean)[-10:]
plt.barh(np.array(NUM + CAT)[order], imp.importances_mean[order]); plt.title("Top reasons customers leave (feature importance)")
plt.tight_layout(); plt.savefig(f"{args.out}/feature_importance.png", dpi=120); plt.close()
metrics["top_features"] = list(np.array(NUM + CAT)[order][::-1])

# 6) Save model, metrics and the 25 riskiest customers
joblib.dump(best, f"{args.out}/model.joblib")
json.dump(metrics, open(f"{args.out}/metrics.json", "w"), indent=2, default=float)
risky = pd.DataFrame({"customerID": df.loc[X_te.index, "customerID"], "churn_probability": proba.round(3)})
risky.sort_values("churn_probability", ascending=False).head(25).to_csv(f"{args.out}/top_risk_customers.csv", index=False)

# 7) If running on Vertex AI, copy results to Google Cloud Storage
model_dir = os.environ.get("AIP_MODEL_DIR", "")
if model_dir.startswith("gs://"):
    from google.cloud import storage
    bucket, _, prefix = model_dir[5:].partition("/")
    for f in ["model.joblib", "metrics.json", "feature_importance.png"]:
        storage.Client().bucket(bucket).blob(f"{prefix}{f}").upload_from_filename(f"{args.out}/{f}")
    print("Uploaded to", model_dir)
