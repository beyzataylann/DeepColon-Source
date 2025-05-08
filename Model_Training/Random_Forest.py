import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from joblib import dump
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

df = pd.read_csv("/content/drive/MyDrive/bitirmeproje/3000resnetgooglenetlbp.csv")
df['class_name'] = df['class_name'].map({'colon_n': 0, 'colon_aca': 1})

X = df.drop(columns=["image_name", "class_name", "label"]).values
y = df['class_name'].values

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.5, random_state=42)

best_val_f1 = 0
best_model = None
patience = 3
counter = 0
n_estimators_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

print("Training started...\n")
for epoch, n_estimators in enumerate(n_estimators_list, 1):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,                  
        min_samples_split=10,         
        min_samples_leaf=5,            
        max_features='sqrt',           
        class_weight='balanced',       
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    train_log_loss = log_loss(y_train, rf.predict_proba(X_train))

    y_val_pred = rf.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average='macro')
    val_log_loss = log_loss(y_val, rf.predict_proba(X_val))

    print(f"Epoch {epoch} | n_estimators={n_estimators} | Train F1 Macro: {train_f1:.4f} | Train Log Loss: {train_log_loss:.4f} | Validation F1 Macro: {val_f1:.4f} | Validation Log Loss: {val_log_loss:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model = rf
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print(f"Early stopping triggered at Epoch {epoch}, n_estimators={n_estimators}.")
        break


print("\nEvaluating best model on test set:")
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_micro = f1_score(y_test, y_pred, average='micro')

print(f"Test Accuracy     : {acc:.4f}")
print(f"Test F1 Macro     : {f1_macro:.4f}")
print(f"Test F1 Micro     : {f1_micro:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

dump(best_model, "best_random_forest_modelrglbpyaren1.joblib")
print("\nModel kaydedildi: best_random_forest_modelrglbpyaren1.joblib")
