import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()

num_samples_train = 10
num_samples_test = 10

pos_train_idx = np.where(y_train == 0)[0][:num_samples_train]
neg_train_idx = np.where(y_train != 0)[0][:num_samples_train]

pos_test_idx = np.where(y_test == 0)[0][:num_samples_test]
neg_test_idx = np.where(y_test != 0)[0][:num_samples_test]

X_train = np.concatenate((x_train[pos_train_idx], x_train[neg_train_idx]), axis=0)
y_train_bin = np.concatenate((np.ones(len(pos_train_idx)), np.zeros(len(neg_train_idx))), axis=0)

X_test = np.concatenate((x_test[pos_test_idx], x_test[neg_test_idx]), axis=0)
y_test_bin = np.concatenate((np.ones(len(pos_test_idx)), np.zeros(len(neg_test_idx))), axis=0)

train_perm = np.random.permutation(len(y_train_bin))
X_train = X_train[train_perm]
y_train_bin = y_train_bin[train_perm]

test_perm = np.random.permutation(len(y_test_bin))
X_test = X_test[test_perm]
y_test_bin = y_test_bin[test_perm]

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train_bin, epochs=20, batch_size=4, validation_data=(X_test, y_test_bin))
test_loss, test_acc = model.evaluate(X_test, y_test_bin, verbose=0)
print(f'Test Accuracy: {test_acc*100:.2f}%')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()