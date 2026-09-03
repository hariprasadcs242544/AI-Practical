import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
print("Hariprasad Vishwakarma T127")
df = pd.read_csv(r"F:\Downloads\Most Runs - 2016.csv")
print("Dataset Loaded Successfully!")
print("\nDataset Shape:")
print(df.shape)
print("\nFirst 5 Rows:")
print(df.head())

if "Player" in df.columns:
    df = df.drop("Player", axis=1)
if "POS" in df.columns:
    df = df.drop("POS", axis=1)

if "HS" in df.columns:
    df["HS"] = df["HS"].astype(str).str.replace("*", "", regex=False)
    df["HS"] = pd.to_numeric(df["HS"], errors="coerce")

for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df = df.dropna()
print("\nMissing values removed")

def performance_category(runs):
    if runs <= 250:
        return "Low"
    elif runs < 450:
        return "Medium"
    else:
        return "High"

df["Performance_Category"] = df["Runs"].apply(performance_category)

print("\nPerformance Categories:")
print(df["Performance_Category"].value_counts())

X = df.drop(["Runs", "Performance_Category"], axis=1)
y = df["Performance_Category"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("\nClasses:")
for i, name in enumerate(encoder.classes_):
    print(i, "=", name)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTraining Samples:")
print(len(X_train))
print("\nTesting Samples:")
print(len(X_test))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nData Standardized Successfully!")

model = GaussianNB()
model.fit(X_train, y_train)

print("\nGaussian Naive Bayes Model Trained Successfully!")

y_pred = model.predict(X_test)

print("\nPredicted Classes:")
print(encoder.inverse_transform(y_pred))

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)
print("\nAccuracy Percentage:")
print(round(accuracy * 100, 2), "%")

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_,
    zero_division=0
))

y_prob = model.predict_proba(X_test)

print("\nPrediction Probabilities:")
print(y_prob)

y_test_binary = label_binarize(
    y_test,
    classes=range(len(encoder.classes_))
)

# Create Confusion Matrix Figure
plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("Confusion Matrix - IPL 2016 Player Performance")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(range(len(encoder.classes_)), encoder.classes_)
plt.yticks(range(len(encoder.classes_)), encoder.classes_)

for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()

# Create ROC Curve Figure
plt.figure(figsize=(8, 6))

for i in range(len(encoder.classes_)):
    if len(set(y_test_binary[:, i])) == 2:
        fpr, tpr, _ = roc_curve(
            y_test_binary[:, i],
            y_prob[:, i]
        )
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            label=encoder.classes_[i] +
            " (AUC = " +
            str(round(roc_auc, 2)) +
            ")"
        )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Gaussian Naive Bayes")
plt.legend()
plt.grid()
plt.tight_layout()
# Display all graphs
plt.show()
print("\n==============================")
print("NAIVE BAYES PRACTICAL COMPLETED")
print("==============================")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("==============================")
