import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Input, Dropout, Add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog


TF_ENABLE_ONEDNN_OPTS = 0


def select_image():
    # Create a Tkinter root window (it will not be shown)
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(
        title="Select an image file",
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")],
    )

    # If a file was selected
    if file_path:
        print(f"Selected image: {file_path}")
        image = Image.open(file_path)
        image.show()  # Open the image in the default viewer
        return image
    else:
        print("No file selected.")
        return None


def build_resnet_model():
    resnet = ResNet50(weights="imagenet")
    resnet = Model(
        inputs=resnet.input, outputs=resnet.layers[-2].output
    )  # Removing the final classification layer
    return resnet


def extract_image_features(image_path, model):
    image = Image.open(image_path).resize((224, 224))
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)
    features = model.predict(image)
    return features


def preprocess_captions(captions, max_length):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1

    sequences = tokenizer.texts_to_sequences(captions)
    sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

    return sequences, tokenizer, vocab_size


# Example caption preprocessing
captions = [
    "startseq a man riding a horse endseq",
    "startseq a dog running in a field endseq",
]
max_length = max(len(c.split()) for c in captions)
sequences, tokenizer, vocab_size = preprocess_captions(captions, max_length)


def build_captioning_model(
    vocab_size, max_length, feature_dim=2048, embedding_dim=256, units=512
):
    # Image feature input
    image_input = Input(shape=(feature_dim,))
    image_dense = Dense(units, activation="relu")(image_input)
    image_dropout = Dropout(0.5)(image_dense)

    # Caption input
    caption_input = Input(shape=(max_length,))
    caption_embedding = Embedding(vocab_size, embedding_dim)(
        caption_input
    )  # Removed `input_length` argument
    caption_lstm = LSTM(units)(caption_embedding)

    # Combine features and captions
    combined = Add()([image_dropout, caption_lstm])
    dense = Dense(units, activation="relu")(combined)
    output = Dense(vocab_size, activation="softmax")(dense)

    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    return model


# Build the model
captioning_model = build_captioning_model(vocab_size, max_length)
captioning_model.summary()


@tf.function(reduce_retracing=True)
def train_step(captioning_model, image_feature, input_sequence, target_word):
    with tf.GradientTape() as tape:
        predictions = captioning_model([image_feature, input_sequence], training=True)
        loss = tf.keras.losses.categorical_crossentropy(target_word, predictions)
    gradients = tape.gradient(loss, captioning_model.trainable_variables)
    captioning_model.optimizer.apply_gradients(
        zip(gradients, captioning_model.trainable_variables)
    )
    return loss


def train_model(captioning_model, image_features, sequences, max_length, epochs=20):
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(image_features)):
            input_image = np.array([image_features[i]])

            # For each sequence (caption), we iterate through each word
            for j in range(1, len(sequences[i])):
                # Input sequence is everything before the jth word
                input_sequence = sequences[i][:j]
                input_sequence = pad_sequences([input_sequence], maxlen=max_length)

                # Target word is the jth word in the sequence
                target_word = sequences[i][j]
                target_word = to_categorical([target_word], num_classes=vocab_size)

                # Convert to tensor
                input_sequence = tf.convert_to_tensor(input_sequence, dtype=tf.int32)
                target_word = tf.convert_to_tensor(target_word, dtype=tf.float32)

                # Perform a training step
                loss = train_step(
                    captioning_model, input_image, input_sequence, target_word
                )
                epoch_loss += loss.numpy()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(image_features)}")


# Example training loop (with dummy data)
image_features = np.random.random((len(captions), 2048))  # Replace with actual features
train_model(captioning_model, image_features, sequences, max_length)


def generate_caption(model, tokenizer, image_feature, max_length):
    caption = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        caption += " " + word
        if word == "endseq":
            break
    return caption


# Example usage
image_path = "image_path"
resnet_model = build_resnet_model()
image_feature = extract_image_features(image_path, resnet_model)
caption = generate_caption(captioning_model, tokenizer, image_feature, max_length)
print("Generated Caption:", caption)
