import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Hariprasad Vishwakarma T127")
df = pd.read_csv(r"F:\Downloads\Most Runs - 2020.csv")
print("Dataset loaded successfully!")
print("\nOriginal Dataset Shape:")
print(df.shape)

df = df.drop_duplicates()
print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

if "HS" in df.columns:
    df["HS"] = df["HS"].astype(str).str.replace("*", "", regex=False)
    df["HS"] = pd.to_numeric(df["HS"], errors="coerce")

if "Player" in df.columns:
    df = df.drop("Player", axis=1)

if "POS" in df.columns:
    df = df.drop("POS", axis=1)

for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna()
print("\nMissing values removed successfully!")

median_runs = df["Runs"].median()

df["Performance"] = df["Runs"].apply(
    lambda x: 1 if x >= median_runs else 0
)

print("\nMedian Runs:")
print(median_runs)

print("\nTarget Classes:")
print(df["Performance"].value_counts())

X = df.drop(["Runs", "Performance"], axis=1)
y = df["Performance"]

print("\nNumber of Features:")
print(X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTraining Samples:")
print(len(X_train))

print("\nTesting Samples:")
print(len(X_test))

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])

parameters = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__kernel": ["linear", "rbf"]
}

grid_search = GridSearchCV(
    pipeline,
    parameters,
    cv=3,
    scoring="accuracy",
    n_jobs=1
)

print("\n==============================")
print("Optimizing SVM Parameters...")
print("Please wait...")
print("==============================")

grid_search.fit(X_train, y_train)

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(round(grid_search.best_score_ * 100, 2), "%")

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

print("\nPrediction completed successfully!")

accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Low Performance", "High Performance"],
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.xticks(
    [0, 1],
    ["Low Performance", "High Performance"]
)

plt.yticks(
    [0, 1],
    ["Low Performance", "High Performance"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar()
plt.tight_layout()
plt.show()

print("\n================================")
print("SVM PRACTICAL COMPLETED")
print("================================")

print("\nFinal Accuracy:")
print(round(accuracy * 100, 2), "%")

print("\nBest Parameters:")
print(grid_search.best_params_)

print("================================")
