import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# --------------------------
# 1. Create a folder to save images
# --------------------------
fig_dir = "./figures"
os.makedirs(fig_dir, exist_ok=True)

# --------------------------
# 2. Read Data
# --------------------------
data_train = pd.read_csv(
    "./dataset/adult.data",
    header=None,
    names=[
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ],
    na_values=" ?",
)

data_test = pd.read_csv(
    "./dataset/adult.test",
    header=None,
    names=[
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income",
    ],
    skiprows=1,
    na_values=" ?",
)

# remove missing values
data_train = data_train.dropna()
data_test = data_test.dropna()

# clear income column
data_train["income"] = data_train["income"].str.strip()
data_test["income"] = data_test["income"].str.replace(".", "", regex=False).str.strip()

# --------------------------
# 3. Divide features and labels
# --------------------------
X_train, y_train = data_train.drop("income", axis=1), data_train["income"]
X_test, y_test = data_test.drop("income", axis=1), data_test["income"]

y_train = y_train.apply(lambda x: 1 if ">50K" in x else 0)
y_test = y_test.apply(lambda x: 1 if ">50K" in x else 0)

# Identify categorical and numerical features
cat_features = X_train.select_dtypes(include="object").columns
num_features = X_train.select_dtypes(exclude="object").columns

# --------------------------
# 4. Preprocessing
# --------------------------
preprocess = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("num", StandardScaler(), num_features),
    ]
)

# --------------------------
# 5. Model Definitions
# --------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, solver="saga"),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

# --------------------------
# 6. Train and Evaluate Models
# --------------------------
for name, clf in models.items():
    pipeline = Pipeline([("preprocess", preprocess), ("model", clf)])

    # training timing
    start_train = time.time()
    pipeline.fit(X_train, y_train)
    end_train = time.time()

    # prediction timing
    start_pred = time.time()
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    end_pred = time.time()

    # print results
    print(f"\n=== {name} ===")
    print(f"训练时间: {end_train - start_train:.3f} 秒")
    print(f"预测时间: {end_pred - start_pred:.3f} 秒")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"ROC Curve - {name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{name.replace(' ','_')}_ROC.png"))
    plt.close()

    # --- PR curve ---
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)
    plt.figure()
    plt.plot(recall, precision, color="purple", lw=2, label=f"AP = {ap:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {name}")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"{name.replace(' ','_')}_PR.png"))
    plt.close()
