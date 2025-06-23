import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, confusion_matrix, accuracy_score, log_loss,
    classification_report, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from joblib import dump

warnings.filterwarnings("ignore")


df = pd.read_csv("/content/drive/MyDrive/bitirmeproje/3000resnetgooglenetlbp.csv")
df['class_name'] = df['class_name'].map({'colon_n': 0, 'colon_aca': 1})


X = df.drop(columns=["image_name", "class_name", "label"]).values
y = df['class_name'].values

# Veriyi %60 train, %20 validation, %20 test olarak böl
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

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
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
class_report = classification_report(y_test, y_pred, target_names=['colon_n', 'colon_aca'])

print(f"Test Accuracy        : {acc:.4f}")
print(f"Test F1 Macro        : {f1_macro:.4f}")
print(f"Test F1 Micro        : {f1_micro:.4f}")
print(f"Test Precision Macro : {precision_macro:.4f}")
print(f"Test Recall Macro    : {recall_macro:.4f}")

print("\nClassification Report (per class):\n")
print(class_report)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['colon_n', 'colon_aca'], yticklabels=['colon_n', 'colon_aca'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()


print("\nGörselleştirme yapılıyor (PCA + t-SNE)...")
pca = PCA(n_components=50).fit_transform(X_test)
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_test_2d = tsne.fit_transform(pca)


plt.figure(figsize=(8, 6))
palette = sns.color_palette("Set1", 2)
sns.scatterplot(x=X_test_2d[:, 0], y=X_test_2d[:, 1], hue=y_test, palette=palette, legend='full')
plt.title('t-SNE Visualization of Test Set')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend(title='Classes', labels=['colon_n', 'colon_aca'])
plt.tight_layout()
plt.show()


dump(best_model, "son1_gorsel_602020.joblib")
print("\nModel kaydedildi: son1_602020.joblib")
