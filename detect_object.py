import argparse
from azure.cognitiveservices.vision.customvision.prediction import (
    CustomVisionPredictionClient,
)
from msrest.authentication import ApiKeyCredentials
import json
from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    """
    Parse argument
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


def main():
    args = parse_args()
    config = json.load(open(args.config, "r"))

    credentials = ApiKeyCredentials(in_headers={"Training-key": config["training_key"]})
    trainer = CustomVisionTrainingClient(config["ENDPOINT"], credentials)
    project_list = trainer.get_projects()
    project_id = {}
    for i in project_list:
        temp = i.as_dict()
        project_id[temp["name"]] = temp["id"]
        project_id[temp["name"]] = temp["id"]

    # Now there is a trained endpoint that can be used to make a prediction
    prediction_credentials = ApiKeyCredentials(
        in_headers={"Prediction-key": config["prediction_key"]}
    )
    predictor = CustomVisionPredictionClient(config["ENDPOINT"], prediction_credentials)

    # Open the sample image and get back the prediction results.
    with open(args.image, "rb") as test_data:
        results = predictor.detect_image(
            project_id[config["project_name"]],
            config["publish_iteration_name"],
            test_data,
        )

    # Display the results.
    img = Image.open(args.image)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(
        "static/TaipeiSansTCBeta-Regular.ttf", size=int(5e-2 * img.size[1])
    )
    for prediction in results.predictions:
        if prediction.probability > 0.5:
            left = prediction.bounding_box.left * img.size[0]
            top = prediction.bounding_box.top * img.size[1]
            right = left + prediction.bounding_box.width * img.size[0]
            bot = top + prediction.bounding_box.height * img.size[1]
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


if __name__ == "__main__":
    main()
