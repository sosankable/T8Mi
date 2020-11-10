import argparse
import glob
import json
import os
import time
from azure.cognitiveservices.vision.customvision.training import (
    CustomVisionTrainingClient,
)
from azure.cognitiveservices.vision.customvision.training.models import (
    ImageFileCreateBatch,
    ImageFileCreateEntry,
)
from msrest.authentication import ApiKeyCredentials


def parse_args():
    """
    Parse argument
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_folder", help="image folder", type=str)
    parser.add_argument(
        "-c",
        "--config",
        help="cofigure file path",
        type=str,
        default="image_classification_config.json",
    )
    args = parser.parse_args()
    return args


def add_image(tag, base_image_url, project_id, trainer):
    image_list = []
    image_tag = trainer.create_tag(project_id, tag)
    filenames = glob.glob(os.path.join(base_image_url, tag, "*.jpg"))
    for file_name in filenames:
        with open(file_name, "rb") as image_contents:
            image_list.append(
                ImageFileCreateEntry(
                    name=file_name,
                    contents=image_contents.read(),
                    tag_ids=[image_tag.id],
                )
            )
    return image_list


def main():
    args = parse_args()
    config = json.load(open(args.config, "r"))
    credentials = ApiKeyCredentials(in_headers={"Training-key": config["training_key"]})
    trainer = CustomVisionTrainingClient(config["ENDPOINT"], credentials)
    # Create a new project

    print("Creating project...")
    try:
        project = trainer.create_project(config["project_name"])
        project_id = project.id
    except:
        project_list = trainer.get_projects()
        projects = {}
        for i in project_list:
            temp = i.as_dict()
            projects[temp["name"]] = temp["id"]
            projects[temp["name"]] = temp["id"]
        project_id = projects[config["project_name"]]
    publish_iteration_name = config["publish_iteration_name"]
    prediction_resource_id = config["prediction_resource_id"]
    # Make two tags in the new project

    base_image_url = args.image_folder

    print("Adding images...")
    image_list = []
    for i in config["tag"]:
        image_list += add_image(i, base_image_url, project_id, trainer)

    upload_result = trainer.create_images_from_files(
        project_id, ImageFileCreateBatch(images=image_list)
    )
    if not upload_result.is_batch_successful:
        print("Image batch upload failed.")
        for image in upload_result.images:
            print("Image status: ", image.status)

    print("Training...")
    iteration = trainer.train_project(project_id)
    while iteration.status != "Completed":
        iteration = trainer.get_iteration(project_id, iteration.id)
        print("Training status: " + iteration.status)
        time.sleep(1)

    # The iteration is now trained. Publish it to the project endpoint
    trainer.publish_iteration(
        project_id, iteration.id, publish_iteration_name, prediction_resource_id
    )
    print("Done!")


if __name__ == "__main__":
    main()
