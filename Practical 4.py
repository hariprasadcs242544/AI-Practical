import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load IPL Most Runs dataset
print("Hariprasad Vishwakarma T127")
df = pd.read_csv(r"F:\Downloads\Most Runs - 2020.csv")
print("Dataset Loaded Successfully!")
print("\nDataset Shape:")
print(df.shape)
print("\nFirst 5 Rows:")
print(df.head())
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
print("\nMissing values removed")
# Create performance categories using Runs
def performance_category(runs):
    if runs <= 250:
        return "Low"
    elif runs < 450:
        return "Medium"
    else:
        return "High"
df["Performance_Category"] = df["Runs"].apply(performance_category)
# Separate input and output
X = df.drop(["Runs", "Performance_Category"], axis=1)
y = df["Performance_Category"]
# Convert target classes into numerical values
encoder = LabelEncoder()
y = encoder.fit_transform(y)
print("\nClasses:")
for i, name in enumerate(encoder.classes_):
    print(i, "=", name)
# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("\nTraining Samples:")
print(len(X_train))
print("\nTesting Samples:")
print(len(X_test))
# Standardize input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create Feed Forward Neural Network
model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)
# Train the neural network
model.fit(X_train, y_train)
print("\nNeural Network trained successfully!")

# Prediction
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:")
print(accuracy)
print("\nAccuracy Percentage:")
print(round(accuracy * 100, 2), "%")
# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=encoder.classes_,
    zero_division=0
))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
# Display Confusion Matrix
plt.figure(figsize=(7, 6))
plt.imshow(cm)
plt.title("Confusion Matrix - IPL Player Performance")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks(range(len(encoder.classes_)), encoder.classes_)
plt.yticks(range(len(encoder.classes_)), encoder.classes_)
# Display values inside matrix
for i in range(len(cm)):
    for j in range(len(cm)):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.colorbar()
plt.show()
# Training Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(model.loss_curve_)
plt.title("Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid()
plt.show()
