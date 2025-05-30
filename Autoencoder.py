"""
https://www.kaggle.com/mlg-ulb/creditcardfraud
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score)
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

tf.config.optimizer.set_jit(True)

tf.random.set_seed(42)
np.random.seed(42)

df = pd.read_csv("creditcard.csv")
df.drop(['Time'], axis=1, inplace=True)

scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])

df_train = df[df['Class'] == 0]
df_test = df.copy()

X_train = df_train.drop("Class", axis=1).values
X_test = df_test.drop("Class", axis=1).values
y_test = df_test["Class"].values

input_dim = X_train.shape[1]
encoding_dim = 14

input_layer = Input(shape=(input_dim,))
encoded = Dense(encoding_dim, activation='relu')(input_layer)
encoded = Dense(encoding_dim // 2, activation='relu')(encoded)
decoded = Dense(encoding_dim // 2, activation='relu')(encoded)
decoded = Dense(encoding_dim, activation='relu')(decoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min')
history = autoencoder.fit(X_train, X_train,
                          epochs=20,
                          batch_size=128,
                          shuffle=True,
                          validation_split=0.1,
                          verbose=1,
                          callbacks=[early_stopping])

X_test_pred = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - X_test_pred, 2), axis=1)

X_train_pred = autoencoder.predict(X_train)
mse_train = np.mean(np.power(X_train - X_train_pred, 2), axis=1)

percentile_threshold = np.percentile(mse_train, 95)

fpr, tpr, roc_thresholds = roc_curve(y_test, mse)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = roc_thresholds[optimal_idx]

thresholds = np.linspace(min(mse), max(mse), 100)
best_f1 = 0
best_threshold = percentile_threshold
for t in thresholds:
    y_pred_temp = (mse > t).astype(int)
    current_f1 = f1_score(y_test, y_pred_temp)
    if current_f1 > best_f1:
        best_f1 = current_f1
        best_threshold = t

print("Percentile Threshold (95th):", percentile_threshold)
print("Optimal Threshold (ROC):", optimal_threshold)
print("Optimal Threshold (F1):", best_threshold, "with F1 Score:", best_f1)

final_threshold = best_threshold
y_pred = (mse > final_threshold).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, mse)
print("\nROC AUC Score:", roc_auc)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label="ROC Curve (area = %0.4f)" % roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Fraud Detection")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()

plt.figure(figsize=(8,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Evolution")
plt.legend()
plt.savefig("loss_curve.png")
plt.close()
