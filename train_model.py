import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load data
X = pd.read_csv("model/features.csv")
y = pd.read_csv("model/targets.csv")

# Split data jadi train dan test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Inisialisasi dan latih Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train.values.ravel())

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"📊 Akurasi model: {accuracy * 100:.2f}%")
print("\n🧾 Classification Report:\n")
print(classification_report(y_test, y_pred))

# Simpan model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/decision_tree_model.pkl")
print("\n✅ Model berhasil disimpan di model/decision_tree_model.pkl")