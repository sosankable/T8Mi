"""
Predict mnist data with Azure machine learning
"""
import argparse
import gzip
import json
import os
import requests
import numpy as np
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
    parser.add_argument("-e", "--endpoint_url", type=str, help="Endpoint url")
    args = parser.parse_args()
    return args


def main():
    """
    Predict mnist data with Azure machine learning
    """
    args = parse_args()
    # Prepare the testing data
    test_image = load_image(os.path.join(args.data_folder, "t10k-images-idx3-ubyte.gz"))
    test_image /= 255
    testing_num = np.random.randint(low=0, high=len(test_image) - 1)

    data = {"data": test_image[testing_num].tolist()}
    # Convert to JSON string
    input_data = json.dumps(data)

    # Set the content type
    headers = {"Content-Type": "application/json"}

    # Make the request and display the response
    resp = requests.post(args.endpoint_url, input_data, headers=headers)

    ans = resp.text.replace("[", "").replace("]", "").split(", ")
    ans = int(float(ans[0]))
    print("The answer is {}".format(ans))
    array = np.reshape(test_image[testing_num] * 255, (28, 28))
    img = Image.fromarray(array)
    img.show()
    img = img.convert("RGB")
    img.save("output.png")


if __name__ == "__main__":
    main()