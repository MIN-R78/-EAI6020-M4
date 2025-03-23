from flaml import AutoML
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd

file_path = r""
data = pd.read_csv(file_path)
X = data.drop(columns=["Class"])
y = data["Class"]

# xy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# flaml 
automl = AutoML()

# setting
settings = {
    "time_budget": 60,
    "metric": 'ap',
    "task": 'classification',
    "seed": 42,
    "eval_method": 'holdout',
}

# train
automl.fit(X_train=X_train, y_train=y_train, **settings)

# model
print("Best model:", automl.model.estimator)

y_pred_proba = automl.predict_proba(X_test)[:, 1] 

# Precision-Recall
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
average_precision = average_precision_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"Precision-Recall Curve (AP = {average_precision:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show()

# 0.322 today
threshold = 0.322
y_pred = (y_pred_proba >= threshold).astype(int)


from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")