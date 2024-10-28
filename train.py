import os
import json

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    GlobalAveragePooling2D,
    Dense,
)
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import Sequence


class CustomDataGenerator(Sequence):
    """
    Custom data generator for Keras models.
    """

    def __init__(
        self,
        image_paths,
        scalars,
        batch_size,
        target_size=(256, 256),
        shuffle=True,
    ):
        """
        Initialize the data generator.
        """
        self.image_paths = image_paths
        self.scalars = scalars
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_paths))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """
        Return the number of batches per epoch.
        """
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Generate one batch of data.
        """
        batch_indices = self.indices[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_image_paths = [self.image_paths[i] for i in batch_indices]
        batch_scalars = [self.scalars[i] for i in batch_indices]

        X = np.zeros(
            (len(batch_image_paths), *self.target_size, 3), dtype=np.float32
        )
        Y_image = np.zeros( #HSV Masked images
            (len(batch_image_paths), *self.target_size, 3), dtype=np.float32
        )
        Y_scalar = np.zeros(len(batch_image_paths), dtype=np.float32)

        for i, image_path in enumerate(batch_image_paths):
            img = cv2.imread(image_path)
            img = cv2.resize(img, self.target_size)
            X[i] = img / 255.0
            Y_image[i] = img / 255.0
            Y_scalar[i] = batch_scalars[i]  # Use pre-stored scalar

        return X, [Y_image, Y_scalar]

    def on_epoch_end(self):
        """
        Shuffle indices after each epoch.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)


# Load data from JSON file
with open("path_scalars.json", "r") as file:
    data = json.load(file)

assert len(data["image_paths"]) == len(data["scalars"])

# Split data into training and testing sets
train_paths, test_paths, train_scalars, test_scalars = train_test_split(
    data["image_paths"],
    data["scalars"],
    test_size=0.2,
    random_state=42,
)

batch_size = 32

train_gen = CustomDataGenerator(
    train_paths, train_scalars, batch_size=batch_size
)
test_gen = CustomDataGenerator(
    test_paths, test_scalars, batch_size=batch_size
)

# Define a callback for model checkpointing
checkpoint_callback = ModelCheckpoint(
    'BOWELPREP_SCALAR_{epoch:02d}.h5',
    save_best_only=False,
    save_weights_only=False,
    mode='auto',
    save_freq='epoch',
)

# Define the autoencoder with multi-output
input_img = Input(shape=(256, 256, 3))

# Encoding layers
x = Conv2D(128, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# Decoding layers
x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Scalar prediction
scalar = GlobalAveragePooling2D()(encoded)
scalar_output = Dense(1, activation='linear')(scalar)

# Construct the model
autoencoder = Model(input_img, [decoded, scalar_output])
autoencoder.compile(optimizer=Adam(), loss=['mse', 'mse'])

# Train the model
autoencoder.fit(
    train_gen,
    epochs=50,
    verbose=2,
    steps_per_epoch=len(train_gen),
    validation_data=test_gen,
    validation_steps=len(test_gen),
    callbacks=[checkpoint_callback],
)
