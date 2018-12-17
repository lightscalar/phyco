from config import *
from data_manager import *

from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from tqdm import tqdm


class CNN:
    """Implements a convolutional neural network for algae classification."""

    def __init__(
        self,
        model_name,
        nb_classes=11,
        batch_size=64,
        image_size=128,
        epochs_per_iter=10,
        nb_iter=100,
        load_existing_model=False,
    ):
        """Save CNN basic parameters."""
        self.model_name = model_name
        self.model_location = f"{MODEL_LOCATION}/{model_name}.h5"
        self.image_size = image_size
        self.nb_iter = nb_iter
        self.epochs_per_iter = epochs_per_iter
        self.nb_classes = nb_classes
        if load_existing_model and len(glob(self.model_location)) > 0:
            print("> Loading model.")
            self.model = load_model(self.model_location)
        else:
            self.build_model()

    def build_model(self):
        """Build the convolutional neural network."""
        image_size = self.image_size
        model = Sequential()
        model.add(
            Conv2D(32, (3, 3), padding="same", input_shape=(image_size, image_size, 1))
        )
        model.add(Activation("relu"))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes))
        model.add(Activation("softmax"))
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["acc"]
        )
        self.model = model

    def train(self):
        for itr in tqdm(range(self.nb_iter)):
            # Grab a subset of the total data.
            X, y = grab_a_batch(batch_size=128)
            X = np.expand_dims(X, axis=3)
            self.model.fit(X, y, epochs=self.epochs_per_iter, validation_split=0.1)
            self.model.save(self.model_location)


if __name__ == "__main__":

    # Build and train a network.
    cnn = CNN("algae_classifier", load_existing_model=False)
    cnn.train()
