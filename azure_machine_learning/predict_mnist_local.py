"""
Predict mnist with local model
"""
import argparse
import os
import gzip
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


def load_image(path):
    """
    Load mnist images
    """
    f = gzip.open(path, "r")
    image_size = 28
    f.read(16)
    buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(int(data.shape[0] / 28 / 28), image_size, image_size, 1)
    f.close()
    return data


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data_folder", type=str, help="Path to the training data"
    )
    parser.add_argument("-m", "--model", type=str, help="Path of model")
    args = parser.parse_args()
    return args


def main():
    """
    Predict mnist with local model
    """
    args = parse_args()
    # Prepare the testing data
    test_image = load_image(os.path.join(args.data_folder, "t10k-images-idx3-ubyte.gz"))
    test_image /= 255
    testing_num = np.random.randint(low=0, high=len(test_image) - 1)

    # Load the model and predict
    model = load_model(args.model)
    predict = model.predict(test_image[testing_num].reshape(1, 28, 28, 1))
    # Output the prediction and show the testing image
    print(np.argmax(predict))
    array = np.reshape(test_image[testing_num] * 255, (28, 28))
    img = Image.fromarray(array)
    img.show()
    img = img.convert("RGB")
    img.save("output.png")


if __name__ == "__main__":
    main()
