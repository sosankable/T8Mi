"""
Object detection and image description on LINE bot
"""
import os
import re
import json
import requests
from flask import Flask, request, abort
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    TextSendMessage,
    FlexSendMessage,
    ImageMessage,
)
from imgur_python import Imgur
from PIL import Image, ImageDraw, ImageFont
import time

app = Flask(__name__)


CONFIG = json.load(open("/home/config.json", "r"))

SUBSCRIPTION_KEY = CONFIG["azure"]["subscription_key"]
ENDPOINT = CONFIG["azure"]["endpoint"]
CV_CLIENT = ComputerVisionClient(
    ENDPOINT, CognitiveServicesCredentials(SUBSCRIPTION_KEY)
)

FACE_KEY = CONFIG["azure"]["face_key"]
FACE_END = CONFIG["azure"]["face_end"]
FACE_CLIENT = FaceClient(FACE_END, CognitiveServicesCredentials(FACE_KEY))
PERSON_GROUP_ID = "tibame"

LINE_SECRET = CONFIG["line"]["line_secret"]
LINE_TOKEN = CONFIG["line"]["line_token"]
LINE_BOT = LineBotApi(LINE_TOKEN)
HANDLER = WebhookHandler(LINE_SECRET)


IMGUR_CONFIG = CONFIG["imgur"]
IMGUR_CLIENT = Imgur(config=IMGUR_CONFIG)


def azure_describe(url):
    """
    Output azure image description result
    """
    description_results = CV_CLIENT.describe_image(url)
    output = ""
    for caption in description_results.captions:
        output += "'{}' with confidence {:.2f}% \n".format(
            caption.text, caption.confidence * 100
        )
    return output


def azure_ocr(url):
    """
    Azure OCR: get characters from image url
    """
    ocr_results = CV_CLIENT.read(url, raw=True)
    # Get the operation location (URL with an ID at the end) from the response
    operation_location_remote = ocr_results.headers["Operation-Location"]
    # Grab the ID from the URL
    operation_id = operation_location_remote.split("/")[-1]
    # Call the "GET" API and wait for it to retrieve the results
    while True:
        get_handw_text_results = CV_CLIENT.get_read_result(operation_id)
        if get_handw_text_results.status not in ["notStarted", "running"]:
            break
        time.sleep(1)

    # Get detected text
    text = []
    if get_handw_text_results.status == OperationStatusCodes.succeeded:
        for text_result in get_handw_text_results.analyze_result.read_results:
            for line in text_result.lines:
                if len(line.text) <= 8:
                    text.append(line.text)
    # Filter text for Taiwan license plate
    r = re.compile("[0-9A-Z]{2,4}[.-]{1}[0-9A-Z]{2,4}")
    text = list(filter(r.match, text))
    return text[0].replace(".", "-") if len(text) > 0 else ""


def azure_face_recognition(filename):
    """
    Azure face recognition
    """
    img = open(filename, "r+b")
    detected_face = FACE_CLIENT.face.detect_with_stream(
        img, detection_model="detection_01"
    )
    if len(detected_face) != 1:
        return ""
    results = FACE_CLIENT.face.identify([detected_face[0].face_id], PERSON_GROUP_ID)
    if len(results) == 0:
        return "unknown"
    result = results[0].as_dict()
    if len(result["candidates"]) == 0:
        return "unknown"
    if result["candidates"][0]["confidence"] < 0.5:
        return "unknown"
    person = FACE_CLIENT.person_group_person.get(
        PERSON_GROUP_ID, result["candidates"][0]["person_id"]
    )
    return person.name


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
                    [left, abs(top - self.font_size)],
                    "{} {}".format(name, confidence),
                    fill=(255, 0, 0),
                    font=self.fnt,
                )

    def azure_face_detection(self):
        face_api_url = "{}face/v1.0/detect".format(FACE_END)
        headers = {"Ocp-Apim-Subscription-Key": FACE_KEY}
        params = {
            "returnFaceId": "true",
            "returnFaceLandmarks": "false",
            "returnFaceAttributes": "emotion",
        }
        response = requests.post(
            face_api_url, params=params, headers=headers, json={"url": self.url}
        )
        if len(response.json()) > 0:
            for obj in response.json():
                left = obj["faceRectangle"]["left"]
                top = obj["faceRectangle"]["top"]
                right = obj["faceRectangle"]["left"] + obj["faceRectangle"]["width"]
                bot = obj["faceRectangle"]["top"] + obj["faceRectangle"]["height"]
                emotion = max(
                    obj["faceAttributes"]["emotion"],
                    key=obj["faceAttributes"]["emotion"].get,
                )
                confidence = max(obj["faceAttributes"]["emotion"].values())
                self.draw.rectangle(
                    [left, top, right, bot], outline=(255, 0, 0), width=3
                )
                self.draw.text(
                    [left, abs(top - self.font_size)],
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


@app.route("/")
def hello():
    "hello world"
    return "Hello World!!!!!"


@app.route("/callback", methods=["POST"])
def callback():
    """
    LINE bot webhook callback
    """
    # get X-Line-Signature header value
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)
    # pylint: disable=maybe-no-member
    app.logger.info("Request body: " + body)
    print(body)
    try:
        HANDLER.handle(body, signature)
    except InvalidSignatureError:
        print(
            "Invalid signature. Please check your channel access token/channel secret."
        )
        abort(400)
    return "OK"


@HANDLER.add(MessageEvent, message=TextMessage)
def handle_message(event):
    """
    Reply text message
    """
    message = TextSendMessage(text=event.message.text)
    print(event.source.user_id)
    print(event.source.type)
    # print(LINE_BOT.get_room_member_ids(room_id))
    LINE_BOT.reply_message(event.reply_token, message)


@HANDLER.add(MessageEvent, message=ImageMessage)
def handle_content_message(event):
    """
    Reply Image message with results of image description and objection detection
    """
    if isinstance(event.message, ImageMessage):
        print(event.message)
        print(event.source.user_id)
        print(event.message.id)
        filename = "{}.jpg".format(event.message.id)
        message_content = LINE_BOT.get_message_content(event.message.id)
        with open(filename, "wb") as f_w:
            for chunk in message_content.iter_content():
                f_w.write(chunk)
        f_w.close()
        image = IMGUR_CLIENT.image_upload(filename, "first", "first")
        link = image["response"]["data"]["link"]
        name = azure_face_recognition(filename)

        if name != "":
            output = name
        else:
            plate = azure_ocr(link)
            az_output = AzureImageOutput(link, filename)
            link = az_output()
            if len(plate) > 0:
                output = "License Plate: {}".format(plate)
            else:
                output = azure_describe(link)

        with open("templates/detect_result.json", "r") as f_r:
            bubble = json.load(f_r)
        f_r.close()
        bubble["body"]["contents"][0]["contents"][0]["contents"][0]["text"] = output
        bubble["header"]["contents"][0]["contents"][0]["contents"][0]["url"] = link
        LINE_BOT.reply_message(
            event.reply_token, [FlexSendMessage(alt_text="Report", contents=bubble)]
        )
