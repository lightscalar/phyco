from vessel import Vessel

from cv2 import resize
from glob import glob
from ipdb import set_trace as debug
import numpy as np
from pylab import imread, ion, imshow, close, subplot
from tqdm import tqdm


# Load all training data into the memory.
v = Vessel("targets.dat")
nb_classes = len(v.target_names)


def one_hot(class_number, total_number_of_classes):
    """Convert integer class number to the total number of classes."""
    vector = np.zeros(total_number_of_classes)
    vector[class_number] = 1
    return vector


def resize_image(image, target_size=128):
    """Resize an image to specified width, preserving aspect ratio."""
    height, width = image.shape
    h2 = np.int(height / 2)
    w2 = np.int(width / 2)
    max_size = np.min((height, width))
    half_max = np.int(max_size / 2)
    square_image = image[h2 - half_max : h2 + half_max, w2 - half_max : w2 + half_max]
    return resize(square_image, (target_size, target_size))


def grab_a_batch(batch_size=64, training_fraction=0.8):
    """Grab a balanced training data sample set."""
    examples_per_class = int(np.ceil(batch_size / nb_classes))
    X = []
    y = []
    for target_name in v.target_names:
        nb_images = len(v.targets[target_name])
        nb_training_images = int(training_fraction * nb_images)
        batch_idx = np.random.randint(0, int(nb_training_images), examples_per_class)
        for idx in batch_idx:
            X.append(v.X[target_name][idx])
            y.append(v.y[target_name][idx])
    X = np.array(X)
    y = np.array(y)
    return X, y


if __name__ == "__main__":

    v = Vessel("targets.dat")

    # X — images; y — class identities (one-hot vectors).
    v.X = {}
    v.y = {}

    targets = sorted(glob(f"data/*"))
    v.targets = {}
    for target in targets:
        target_name = target.split("/")[-1]
        v.targets[target_name] = glob(f"{target}/*")
    target_names = list(v.targets.keys())
    v.target_names = target_names

    # Now generate training/test data.
    for itr, target in enumerate(tqdm(v.target_names)):
        v.X[target] = []
        v.y[target] = []
        paths_to_images = v.targets[target]
        for path_to_image in paths_to_images:
            # Open image and resize it appropriately.
            image = imread(path_to_image)
            image_ = resize_image(image)
            # Add standardized images and class labels.
            v.X[target].append(image_)
            v.y[target].append(one_hot(itr, len(v.target_names)))
        # Convert to numpy arrays
        v.X[target] = np.array(v.X[target])
        v.y[target] = np.array(v.y[target])
    v.save()
