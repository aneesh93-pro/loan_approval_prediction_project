"""Train and evaluate the Loan Approval Decision Tree pipeline.

Expected input:
    dataset/loan_approval_dataset.csv

Run from the repository root:
    python src/train.py
"""
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "loan_approval_dataset.csv"
ASSETS_DIR = ROOT / "assets"
MODELS_DIR = ROOT / "models"
ASSETS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place loan_approval_dataset.csv inside the dataset/ folder."
    )

df = pd.read_csv(DATA_PATH)
required_columns = {
    "Loan_ID", "Gender", "Married", "Loan_Status"
}
missing = required_columns.difference(df.columns)
if missing:
    raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

df["Loan_Status"] = df["Loan_Status"].astype(str).str.strip()
y = df["Loan_Status"].map({"N": 0, "Y": 1})
if y.isna().any():
    unexpected = sorted(df.loc[y.isna(), "Loan_Status"].unique().tolist())
    raise ValueError(f"Unexpected Loan_Status values: {unexpected}")

# Keep the same feature selection used in the supplied notebook.
X = df.drop(columns=["Loan_ID", "Gender", "Married", "Loan_Status"])
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("numerical", "passthrough", numeric_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            DecisionTreeClassifier(
                criterion="gini",
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
            ),
        ),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
pipeline.fit(X_train, y_train)
train_pred = pipeline.predict(X_train)
test_pred = pipeline.predict(X_test)

train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)
report = classification_report(
    y_test,
    test_pred,
    labels=[0, 1],
    target_names=["Rejected", "Approved"],
    zero_division=0,
    output_dict=True,
)

print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Testing accuracy:  {test_accuracy:.4f}")
print("\nClassification report:")
print(classification_report(
    y_test, test_pred, labels=[0, 1],
    target_names=["Rejected", "Approved"], zero_division=0
))

# Save the trained pipeline and evaluation metadata.
model_path = MODELS_DIR / "decision_tree_project.pkl"
joblib.dump(pipeline, model_path)
metrics = {
    "training_accuracy": train_accuracy,
    "testing_accuracy": test_accuracy,
    "classification_report": report,
    "test_samples": int(len(y_test)),
    "features": X.columns.tolist(),
}
(ASSETS_DIR / "evaluation_metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)

# Target class distribution
fig, ax = plt.subplots(figsize=(7, 5))
y.map({0: "Rejected", 1: "Approved"}).value_counts().reindex(
    ["Rejected", "Approved"]
).plot(kind="bar", rot=0, ax=ax)
ax.set_title("Loan Status Distribution")
ax.set_xlabel("Loan Status")
ax.set_ylabel("Number of Applications")
fig.tight_layout()
fig.savefig(ASSETS_DIR / "class_distribution.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Confusion matrix
cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Rejected", "Approved"],
).plot(ax=ax, cmap="Blues", values_format="d")
ax.set_title("Loan Approval - Confusion Matrix")
fig.tight_layout()
fig.savefig(ASSETS_DIR / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Tree visualization (first four levels for readability)
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
fig, ax = plt.subplots(figsize=(24, 12))
plot_tree(
    pipeline.named_steps["classifier"],
    feature_names=feature_names,
    class_names=["Rejected", "Approved"],
    filled=True,
    rounded=True,
    max_depth=4,
    fontsize=8,
    ax=ax,
)
ax.set_title("Loan Approval Decision Tree")
fig.tight_layout()
fig.savefig(ASSETS_DIR / "decision_tree.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ROC curve and AUC (probability-based evaluation)
from sklearn.metrics import RocCurveDisplay, roc_auc_score

if hasattr(pipeline, "predict_proba"):
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, test_probabilities)
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, test_probabilities, ax=ax, name=f"Decision Tree (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax.set_title("ROC Curve - Loan Approval Classifier")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    metrics["roc_auc"] = float(auc)
    (ASSETS_DIR / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

# Feature importance (top 12 transformed features)
importances = pipeline.named_steps["classifier"].feature_importances_
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values("Importance", ascending=False).head(12).sort_values("Importance")
fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(importance_df["Feature"], importance_df["Importance"])
ax.set_title("Top Feature Importances - Decision Tree")
ax.set_xlabel("Importance")
ax.set_ylabel("Encoded Feature")
fig.tight_layout()
fig.savefig(ASSETS_DIR / "feature_importance.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Numeric feature correlation heatmap for exploratory analysis
numeric_df = X.select_dtypes(include=["number"])
if numeric_df.shape[1] >= 2:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = numeric_df.corr(numeric_only=True)
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Correlation Heatmap - Numerical Features")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# Train vs test accuracy comparison
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Training", "Testing"], [train_accuracy, test_accuracy])
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")
ax.set_title("Training vs Testing Accuracy")
for i, value in enumerate([train_accuracy, test_accuracy]):
    ax.text(i, value + 0.015, f"{value:.1%}", ha="center")
fig.tight_layout()
fig.savefig(ASSETS_DIR / "train_test_accuracy.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# First 20 test predictions in a simple table image
results = pd.DataFrame({
    "Actual": pd.Series(y_test).map({0: "Rejected", 1: "Approved"}).to_numpy(),
    "Predicted": pd.Series(test_pred).map({0: "Rejected", 1: "Approved"}).to_numpy(),
}).head(20)
fig, ax = plt.subplots(figsize=(9, 7))
ax.axis("off")
table = ax.table(
    cellText=results.values,
    colLabels=results.columns,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
ax.set_title("Actual vs Predicted Loan Status", pad=20)
fig.tight_layout()
fig.savefig(ASSETS_DIR / "prediction_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved model: {model_path}")
print(f"Saved visual assets to: {ASSETS_DIR}")
