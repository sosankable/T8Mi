"""
Object Detection with Azure Custom Vision
"""
import json

import argparse
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from msrest.authentication import ApiKeyCredentials
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", help="image path", type=str)
    parser.add_argument(
        "-c",
        "--config",
        help="cofigure file path",
        type=str,
        default="object_detection_config.json",
    )
    args = parser.parse_args()
    return args


def get_project_id(config):
    """
    Get project ID list
    """
    credentials = ApiKeyCredentials(in_headers={"Training-key": config["training_key"]})
    trainer = CustomVisionTrainingClient(config["ENDPOINT"], credentials)
    project_id = next(
        proj.id
        for proj in trainer.get_projects()
        if proj.name == config["project_name"]
    )
    return project_id


def main():
    """
    Object Detection with Azure Custom Vision
    """
    args = parse_args()
    config = json.load(open(args.config, "r"))

    # Get the predictor
    prediction_credentials = ApiKeyCredentials(
        in_headers={"Prediction-key": config["prediction_key"]}
    )
    predictor = CustomVisionPredictionClient(config["ENDPOINT"], prediction_credentials)

    # ======================================================================================
    # Open the sample image and get back the prediction results.
    project_id = get_project_id(config)
    with open(args.image, "rb") as test_data:
        results = predictor.detect_image(
            project_id,
            config["publish_iteration_name"],
            test_data,
        )

    # ======================================================================================
    # Draw the bounding boxes on the image
    img = Image.open(args.image)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(
        "../static/TaipeiSansTCBeta-Regular.ttf", size=int(5e-2 * img.size[1])
    )
    for prediction in results.predictions:
        if prediction.probability > 0.5:
            bbox = prediction.bounding_box.as_dict()
            left = bbox['left'] * img.size[0]
            top = bbox['top'] * img.size[1]
            right = left + bbox['width'] * img.size[0]
            bot = top + bbox['height'] * img.size[1]
            draw.rectangle([left, top, right, bot], outline=(255, 0, 0), width=3)
            draw.text(
                [left, abs(top - 5e-2 * img.size[1])],
                "{0} {1:0.2f}".format(
                    prediction.tag_name, prediction.probability * 100
                ),
                fill=(255, 0, 0),
                font=font,
            )

    img.save("output.png")
    print("Done!")
    print("Please check ouptut.png")


if __name__ == "__main__":
    main()
