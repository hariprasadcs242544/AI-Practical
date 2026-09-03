print("hariprasad vishwakarma T127")
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load the dataset
df = pd.read_csv(r"F:\Downloads\Most Runs - 2018.csv")
print("First 5 rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
# Remove unnecessary columns
if "Player" in df.columns:
    df = df.drop("Player", axis=1)
if "POS" in df.columns:
    df = df.drop("POS", axis=1)
# Clean Highest Score column
if "HS" in df.columns:
    df["HS"] = df["HS"].astype(str).str.replace("*", "", regex=False)
    df["HS"] = pd.to_numeric(df["HS"], errors="coerce")
# Convert columns to numeric
for column in df.columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")
# Remove missing values
df = df.dropna()
# Create Performance category
median_runs = df["Runs"].median()

print("\nMedian Runs:", median_runs)

df["Performance"] = df["Runs"].apply(
    lambda x: "High" if x >= median_runs else "Low"
)
print("\nPerformance Distribution:")
print(df["Performance"].value_counts())
# Separate input and output
# Remove Runs because Performance was created from Runs
X = df.drop(["Performance", "Runs"], axis=1)
y = df["Performance"]
print("\nNumber of Features:", X.shape[1])
print("Number of Samples:", X.shape[0])
# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])


# Create Decision Tree
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)
# Train the model
model.fit(X_train, y_train)
print("\nDecision Tree trained successfully!")
# Prediction
y_pred = model.predict(X_test)
print("\nPredicted Values:")
print(y_pred)
# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")
# Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Low", "High"]
    )
)
# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Decision Tree Visualization
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Low", "High"],
    filled=True,
    rounded=True
)
plt.title("Decision Tree - Cricket Player Performance Classification")

plt.show()
