"""
Training for object detection with Azure Custom Vision
"""
import argparse
import json
import os
import time
from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from azure.cognitiveservices.vision.customvision.training.models import (
    ImageFileCreateBatch,
    ImageFileCreateEntry,
    Region,
)
from msrest.authentication import ApiKeyCredentials


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        help="cofigure file path",
        type=str,
        default="object_detection_config.json",
    )
    args = parser.parse_args()
    return args


def add_image(trainer, label, project_id, annotation, image_folder):
    """
    Add images with labels and bounding box
    """
    tagged_images_with_regions = []
    tag = trainer.create_tag(project_id, label)
    for file_name in annotation.keys():
        left, top, width, height = annotation[file_name]
        regions = [
            Region(tag_id=tag.id, left=left, top=top, width=width, height=height)
        ]
        file_path = os.path.join(image_folder, label, file_name + ".jpg")
        with open(file_path, "rb") as image_contents:
            tagged_images_with_regions.append(
                ImageFileCreateEntry(
                    name=file_name, contents=image_contents.read(), regions=regions
                )
            )
        image_contents.close()
    return tagged_images_with_regions


def main():
    """
    Training for object detection with Azure Custom Vision
    """
    args = parse_args()
    config = json.load(open(args.config, "r"))
    credentials = ApiKeyCredentials(in_headers={"Training-key": config["training_key"]})
    trainer = CustomVisionTrainingClient(config["ENDPOINT"], credentials)

    print("Creating project...")

    # Find the object detection domain
    obj_detection_domain = next(
        domain
        for domain in trainer.get_domains()
        if domain.type == "ObjectDetection" and domain.name == "General"
    )
    project = trainer.create_project(
        config["project_name"], domain_id=obj_detection_domain.id
    )

    # ======================================================================================

    print("Adding images...")
    image_folder = config["image_folder"]
    annotations = json.load(open("annotation.json", "r"))
    tagged_images_with_regions = []
    for label in annotations.keys():
        tagged_images_with_regions += add_image(
            trainer, label, project.id, annotations[label], image_folder
        )

    upload_result = trainer.create_images_from_files(
        project.id, ImageFileCreateBatch(images=tagged_images_with_regions)
    )
    if not upload_result.is_batch_successful:
        print("Image batch upload failed.")
        for image in upload_result.images:
            print("Image status: ", image.status)

    # ======================================================================================
    print("Training...")
    publish_iteration_name = config["publish_iteration_name"]
    prediction_resource_id = config["prediction_resource_id"]
    iteration = trainer.train_project(project.id)
    while iteration.status != "Completed":
        iteration = trainer.get_iteration(project.id, iteration.id)
        print("Training status: " + iteration.status)
        time.sleep(1)

    # The iteration is now trained. Publish it to the project endpoint
    trainer.publish_iteration(
        project.id, iteration.id, publish_iteration_name, prediction_resource_id
    )
    print("Done!")


if __name__ == "__main__":
    main()
