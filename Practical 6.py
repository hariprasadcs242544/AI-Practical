import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Hariprasad Vishwakarma T127")
df = pd.read_csv(r"F:\Downloads\Most Runs - 2020.csv")
print("Dataset loaded successfully!")
print("\nDataset Shape:")
print(df.shape)
print("\nFirst 5 Rows:")
print(df.head())

df = df.dropna()
df = df.drop_duplicates()
print("\nShape after cleaning:")
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

median_runs = df["Runs"].median()

df["Performance"] = df["Runs"].apply(lambda x: 1 if x >= median_runs else 0)

print("\nTarget Classes:")
print("0 = Low Performance")
print("1 = High Performance")

print("\nMedian Runs:")
print(median_runs)

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

weak_classifier = DecisionTreeClassifier(
    max_depth=1,
    random_state=42
)

weak_classifier.fit(X_train, y_train)

weak_prediction = weak_classifier.predict(X_test)

weak_accuracy = accuracy_score(y_test, weak_prediction)

print("\n==============================")
print("INDIVIDUAL WEAK CLASSIFIER")
print("==============================")
print("Accuracy:", round(weak_accuracy * 100, 2), "%")

ada_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

ada_model.fit(X_train, y_train)

ada_prediction = ada_model.predict(X_test)

ada_accuracy = accuracy_score(y_test, ada_prediction)

print("\n==============================")
print("ADABOOST RESULT")
print("==============================")
print("Accuracy:", round(ada_accuracy * 100, 2), "%")

improvement = (ada_accuracy - weak_accuracy) * 100

print("\nImprovement:")
print(round(improvement, 2), "percentage points")

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")
print("Weak Classifier Accuracy:")
print(round(weak_accuracy * 100, 2), "%")
print("AdaBoost Accuracy:")
print(round(ada_accuracy * 100, 2), "%")
print("Improvement:")
print(round(improvement, 2), "percentage points")

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")
print(classification_report(
    y_test,
    ada_prediction,
    target_names=["Low Performance", "High Performance"],
    zero_division=0
))

cm = confusion_matrix(y_test, ada_prediction)

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")
print(cm)

plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("AdaBoost Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks([0, 1], ["Low Performance", "High Performance"])
plt.yticks([0, 1], ["Low Performance", "High Performance"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.show()

models = ["Weak Classifier", "AdaBoost"]
accuracies = [weak_accuracy * 100, ada_accuracy * 100]

plt.figure(figsize=(7, 5))
plt.bar(models, accuracies)
plt.title("Weak Classifier vs AdaBoost")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")

for i, value in enumerate(accuracies):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.ylim(0, 110)
plt.tight_layout()
plt.show()

print("\n==============================")
print("ADABOOST PRACTICAL COMPLETED")
print("==============================")
print("Weak Classifier Accuracy:", round(weak_accuracy * 100, 2), "%")
print("AdaBoost Accuracy:", round(ada_accuracy * 100, 2), "%")
print("Improvement:", round(improvement, 2), "percentage points")
print("==============================")
