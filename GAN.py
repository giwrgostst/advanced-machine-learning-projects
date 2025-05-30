"""
https://www.kaggle.com/jangedoo/utkface-new
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

tf.config.optimizer.set_jit(True)

tf.random.set_seed(42)
np.random.seed(42)

IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 3
BATCH_SIZE = 64
LATENT_DIM = 100
NUM_CLASSES = 5
EPOCHS = 40

def map_age_to_group(age):
    if age <= 20:
        return 0
    elif age <= 35:
        return 1
    elif age <= 55:
        return 2
    elif age <= 65:
        return 3
    else:
        return 4

def process_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = (image / 127.5) - 1.0
    filename = tf.strings.split(file_path, os.sep)[-1]
    age_str = tf.strings.split(filename, "_")[0]
    age = tf.strings.to_number(age_str, out_type=tf.int32)
    age_group = tf.py_function(func=lambda a: np.array(map_age_to_group(a)), inp=[age], Tout=tf.int32)
    age_group.set_shape(())
    return image, age_group

dataset_path = os.path.join("data", "UTKFace")
file_pattern = os.path.join(dataset_path, "*.jpg")
files = tf.data.Dataset.list_files(file_pattern, shuffle=True)
dataset = files.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.shuffle(6000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_generator():
    noise_input = layers.Input(shape=(LATENT_DIM,))
    label_input = layers.Input(shape=(1,), dtype='int32')
    label_embedding = layers.Embedding(input_dim=NUM_CLASSES, output_dim=50)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    combined_input = layers.Concatenate()([noise_input, label_embedding])
    x = layers.Dense(4 * 4 * 512, use_bias=False)(combined_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((4, 4, 512))(x)
    x = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', use_bias=False, activation='tanh')(x)
    model = Model([noise_input, label_input], x, name="generator")
    return model

def build_discriminator():
    image_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    label_input = layers.Input(shape=(1,), dtype='int32')
    label_embedding = layers.Embedding(input_dim=NUM_CLASSES, output_dim=IMG_HEIGHT * IMG_WIDTH)(label_input)
    label_embedding = layers.Flatten()(label_embedding)
    label_embedding = layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 1))(label_embedding)
    x = layers.Concatenate(axis=-1)([image_input, label_embedding])
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(512, kernel_size=4, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1)(x)
    model = Model([image_input, label_input], x, name="discriminator")
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)
discriminator_optimizer = Adam(learning_rate=2e-4, beta_1=0.5)

@tf.function
def train_step(real_images, real_labels):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    random_labels = tf.random.uniform([BATCH_SIZE, 1], minval=0, maxval=NUM_CLASSES, dtype=tf.int32)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, random_labels], training=True)
        real_labels_expanded = tf.expand_dims(real_labels, axis=-1)
        real_output = discriminator([real_images, real_labels_expanded], training=True)
        fake_output = discriminator([generated_images, random_labels], training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def generate_and_save_images(model, epoch, test_input=None, test_labels=None):
    if test_input is None:
        latent = tf.random.normal([1, LATENT_DIM])
        test_input = tf.repeat(latent, repeats=NUM_CLASSES, axis=0)
    if test_labels is None:
        test_labels = tf.convert_to_tensor([[i] for i in range(NUM_CLASSES)], dtype=tf.int32)
    predictions = model([test_input, test_labels], training=False)
    predictions = (predictions + 1) / 2.0
    predictions = tf.cast(predictions, tf.float32).numpy()
    fig = plt.figure(figsize=(NUM_CLASSES, 1))
    for i in range(predictions.shape[0]):
        plt.subplot(1, NUM_CLASSES, i + 1)
        plt.imshow(predictions[i])
        plt.axis('off')
    plt.suptitle("Epoch " + str(epoch))
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

def train(dataset, epochs):
    for epoch in range(epochs):
        print("Epoch:", epoch + 1)
        for image_batch, label_batch in dataset:
            g_loss, d_loss = train_step(image_batch, label_batch)
        print("Generator loss: {:.4f} | Discriminator loss: {:.4f}".format(g_loss, d_loss))
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1)

train(dataset, EPOCHS)