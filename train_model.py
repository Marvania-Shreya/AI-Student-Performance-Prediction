import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("student_performance_600.csv")

# -----------------------------
# Feature Selection
# -----------------------------

features = [
    "Attendance",
    "QuizScore",
    "AssignmentScore",
    "InternalExam",
    "PreviousPercentage",
    "Participation",
    "HomeworkCompletion",
    "BehaviourScore",
    "StudyHours",
    "ParentMeeting"
]

X = df[features]

y = df["RiskLevel"]

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train Model
# -----------------------------

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# -----------------------------
# Prediction
# -----------------------------

predictions = model.predict(X_test)

# -----------------------------
# Accuracy
# -----------------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

print("="*40)
print("Model Accuracy")
print("="*40)

print(f"Accuracy : {accuracy:.2f}")

# -----------------------------
# Save Model
# -----------------------------

os.makedirs(
    "model",
    exist_ok=True
)

with open(
    "model/student_model.pkl",
    "wb"
) as file:

    pickle.dump(
        model,
        file
    )

print("="*40)
print("Model Saved Successfully")
print("Location : model/student_model.pkl")
print("="*40)