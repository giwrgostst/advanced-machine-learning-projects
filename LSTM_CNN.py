"""Dataset URL: https://www.kaggle.com/datasets/adityajn105/flickr8k"""

import os
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Embedding, LSTM, add
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.data import AUTOTUNE

physical_devices = tf.config.list_physical_devices('GPU')
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

tf.config.optimizer.set_jit(True)

tf.random.set_seed(42)

def load_images(directory, target_size=(32, 32)):
    images = {}
    for img_name in os.listdir(directory):
        filename = os.path.join(directory, img_name)
        try:
            image = load_img(filename, target_size=target_size)
        except Exception as e:
            print(f"Πρόβλημα με το άνοιγμα της εικόνας {filename}: {e}")
            continue
        image = img_to_array(image)
        image = image.astype('float32') / 255.0
        image_id = os.path.splitext(img_name)[0]
        images[image_id] = image
    return images

def load_descriptions(filename):
    descriptions = {}
    with open(filename, 'r', encoding='utf-8') as file:
        next(file)
        for line in file:
            tokens = line.strip().split(',')
            if len(tokens) < 2:
                continue
            image_id, caption = tokens[0], tokens[1]
            image_id = os.path.splitext(image_id)[0]
            if image_id not in descriptions:
                descriptions[image_id] = []
            caption = caption.lower()
            caption = caption.translate(str.maketrans('', '', string.punctuation))
            caption = 'startseq ' + caption + ' endseq'
            descriptions[image_id].append(caption)
    return descriptions

def clean_descriptions(descriptions):
    all_captions = []
    for key in descriptions.keys():
        all_captions.extend(descriptions[key])
    return all_captions

def create_tokenizer(descriptions):
    lines = clean_descriptions(descriptions)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_length(descriptions):
    lines = clean_descriptions(descriptions)
    return max(len(caption.split()) for caption in lines)

def create_sequences(tokenizer, max_length, descriptions, images, vocab_size, keys):
    X1, X2, y = [], [], []
    for key in keys:
        if key not in images:
            continue
        image = images[key]
        for desc in descriptions[key]:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(image)
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

def define_model(vocab_size, max_length, image_shape=(32, 32, 3)):
    image_input = Input(shape=image_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_initializer='glorot_uniform')(x)
    
    caption_input = Input(shape=(max_length,))
    se = Embedding(vocab_size, 256, mask_zero=True, embeddings_initializer='glorot_uniform')(caption_input)
    se = Dropout(0.5)(se)
    se = LSTM(256, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
              dropout=0.5, recurrent_dropout=0.5)(se)
    
    decoder = add([x, se])
    decoder = Dense(256, activation='relu', kernel_initializer='glorot_uniform')(decoder)
    outputs = Dense(vocab_size, activation='softmax', kernel_initializer='glorot_uniform')(decoder)
    
    model = Model(inputs=[image_input, caption_input], outputs=outputs)
    optimizer = Adam(clipnorm=5.0)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, image, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([np.expand_dims(image, axis=0), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

if __name__ == '__main__':
    dataset_images_path = r'archive/Images'
    dataset_captions_file = r'archive/captions.txt'
    NUM_TRAIN_IMAGES = 500
    NUM_TEST_IMAGES = 100

    all_images = load_images(dataset_images_path, target_size=(32, 32))
    all_descriptions = load_descriptions(dataset_captions_file)
    
    all_keys = list(all_descriptions.keys())
    train_keys = all_keys[:NUM_TRAIN_IMAGES]
    test_keys = all_keys[NUM_TRAIN_IMAGES:NUM_TRAIN_IMAGES+NUM_TEST_IMAGES]
    
    train_descriptions = {key: all_descriptions[key] for key in train_keys if key in all_descriptions}
    train_images = {key: all_images[key] for key in train_keys if key in all_images}
    
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    max_len = max_length(train_descriptions)
    print(f"Μέγιστο μήκος λεζάντας: {max_len}")
    print(f"Μέγεθος λεξιλογίου: {vocab_size}")
    
    X_img, X_seq, y = create_sequences(tokenizer, max_len, train_descriptions, train_images, vocab_size, train_keys)
    print(f"Training data: Εικόνες: {X_img.shape}, Ακολουθίες: {X_seq.shape}, Έξοδοι: {y.shape}")
    
    BATCH_SIZE = 128
    def process_data(inputs, labels):
        return inputs, labels

    dataset = tf.data.Dataset.from_tensor_slices(((X_img, X_seq), y))
    dataset = dataset.map(process_data, num_parallel_calls=AUTOTUNE)
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    model = define_model(vocab_size, max_len, image_shape=(32, 32, 3))
    model.summary()
    model.fit(dataset, epochs=20, verbose=2)
    
    model.save('image_caption_model_500Images.h5')
    print("Το μοντέλο αποθηκεύτηκε στο 'image_caption_model_500Images.h5'")
    
    print("\nΠαραγωγή λεζάντας για τις εικόνες του test set:")
    for image_id in test_keys:
        if image_id not in all_images:
            continue
        caption = generate_desc(model, tokenizer, all_images[image_id], max_len)
        print("Image:", image_id)
        print("Caption:", caption)
        print("-" * 50)
