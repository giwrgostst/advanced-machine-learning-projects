"""
Dataset: https://www.tensorflow.org/datasets/catalog/cats_vs_dogs
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU memory growth enabled.")
else:
    print("No GPU found, continuing on CPU.")

tf.config.optimizer.set_jit(True)
print("XLA JIT compilation enabled.")

IMG_SIZE = 180
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

resize_and_rescale = keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1.0/255)
], name='resize_and_rescale')

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
], name='data_augmentation')

(train_ds, val_ds, test_ds), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    as_supervised=True,
    with_info=True
)
print("Dataset loaded with {} classes.".format(metadata.features['label'].num_classes))

def prepare(ds, shuffle=False):
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.padded_batch(BATCH_SIZE, padded_shapes=([None, None, 3], []))
    return ds.prefetch(buffer_size=AUTOTUNE)

train_ds_prepared = prepare(train_ds, shuffle=True)
val_ds_prepared = prepare(val_ds)
test_ds_prepared = prepare(test_ds)

model = keras.Sequential([
    layers.Input(shape=(None, None, 3)),
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(metadata.features['label'].num_classes, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 3
history = model.fit(
    train_ds_prepared,
    validation_data=val_ds_prepared,
    epochs=EPOCHS
)

plt.figure(figsize=(10, 8))
sample_batch, _ = next(iter(train_ds.padded_batch(6, padded_shapes=([None, None, 3], []))))
augmented_images = data_augmentation(resize_and_rescale(sample_batch), training=True)
augmented_images = np.clip(augmented_images, 0.0, 1.0)
for i in range(6):
    ax = plt.subplot(2, 3, i+1)
    plt.imshow(augmented_images[i])
    plt.axis("off")
plt.suptitle("Examples of Augmented Images", fontsize=16)
plt.savefig("augmented_images.png")
plt.show()

def generate_sine_wave(seq_length=1000, freq=0.05):
    t = np.arange(seq_length)
    sine_wave = np.sin(2 * np.pi * freq * t)
    return sine_wave

original_series = generate_sine_wave()

def add_noise(series, noise_factor=0.1):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=series.shape)
    return series + noise

def scale_series(series, scale_factor=1.2):
    return series * scale_factor

def time_shift(series, shift_max=50):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(series, shift)

def time_warp(series, warp_factor=0.1):
    seq_length = len(series)
    original_indices = np.arange(seq_length)
    new_length = int(seq_length * (1 + warp_factor))
    new_indices = np.linspace(0, seq_length - 1, num=new_length)
    warped_series = np.interp(original_indices, new_indices, np.interp(new_indices, original_indices, series))
    return warped_series[:seq_length]

aug_series_noise = add_noise(original_series)
aug_series_scaled = scale_series(original_series)
aug_series_shifted = time_shift(original_series)
aug_series_warped = time_warp(original_series)

plt.figure(figsize=(12, 8))
plt.plot(original_series, label="Original")
plt.plot(aug_series_noise, label="With Noise")
plt.plot(aug_series_scaled, label="Scaled")
plt.plot(aug_series_shifted, label="Shifted")
plt.plot(aug_series_warped, label="Time Warped")
plt.legend()
plt.title("Time Series Data Augmentation")
plt.xlabel("Time")
plt.ylabel("Value")
plt.savefig("time_series_aug.png")
plt.show()
