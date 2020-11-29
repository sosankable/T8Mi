import os
import json
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image, ImageDraw, ImageFont
from imgur_python import Imgur

CONFIG = json.load(open("/home/config.json", "r"))

SUBSCRIPTION_KEY = CONFIG["azure"]["subscription_key"]
ENDPOINT = CONFIG["azure"]["endpoint"]
CV_CLIENT = ComputerVisionClient(
    ENDPOINT, CognitiveServicesCredentials(SUBSCRIPTION_KEY)
)

FACE_KEY = CONFIG["azure"]["face_key"]
FACE_END = CONFIG["azure"]["face_end"]
FACE_CLIENT = FaceClient(FACE_END, CognitiveServicesCredentials(FACE_KEY))
IMGUR_CONFIG = CONFIG["imgur"]
IMGUR_CLIENT = Imgur(config=IMGUR_CONFIG)


class AzureImageOutput:
    def __init__(self, url, filename):
        self.url = url
        self.filename = filename
        self.img = Image.open(filename)
        self.draw = ImageDraw.Draw(self.img)
        self.font_size = int(5e-2 * self.img.size[1])
        self.fnt = ImageFont.truetype(
            "static/TaipeiSansTCBeta-Regular.ttf", size=self.font_size
        )

    def azure_object_detection(self):
        object_detection = CV_CLIENT.detect_objects(self.url)
        if len(object_detection.objects) > 0:
            for obj in object_detection.objects:
                left = obj.rectangle.x
                top = obj.rectangle.y
                right = obj.rectangle.x + obj.rectangle.w
                bot = obj.rectangle.y + obj.rectangle.h
                name = obj.object_property
                confidence = obj.confidence
                print(
                    "{} at location {}, {}, {}, {}".format(name, left, right, top, bot)
                )
                self.draw.rectangle(
                    [left, top, right, bot], outline=(255, 0, 0), width=3
                )
                self.draw.text(
                    [left, top + self.font_size],
                    "{} {}".format(name, confidence),
                    fill=(255, 0, 0),
                    font=self.fnt,
                )

    def azure_face_detection(self):

        detected_faces = FACE_CLIENT.face.detect_with_url(
            url=self.url,
            detectionModel="detection_02",
            return_recognition_model=True,
            return_face_landmarks=True,
            return_face_attributes=["emotion"],
        )
        if len(detected_faces) > 0:
            for face in detected_faces:
                rectangle = face.face_rectangle.as_dict()
                bbox = [
                    rectangle["left"],
                    rectangle["top"],
                    rectangle["left"] + rectangle["width"],
                    rectangle["top"] + rectangle["height"],
                ]
                emotions = face.face_attributes.as_dict()["emotion"]
                emotion = max(emotions, key=emotions.get)
                confidence = max(emotions.values())
                self.draw.rectangle(bbox, outline=(255, 0, 0), width=3)
                self.draw.text(
                    [bbox[0], abs(bbox[1] - self.font_size)],
                    "{} {}".format(emotion, confidence),
                    fill=(255, 0, 0),
                    font=self.fnt,
                )

    def __call__(self):
        self.azure_object_detection()
        self.azure_face_detection()
        self.img.save(self.filename)
        image = IMGUR_CLIENT.image_upload(self.filename, "first", "first")
        link = image["response"]["data"]["link"]
        os.remove(self.filename)
        return link
