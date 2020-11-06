"""
Object detection and image description on LINE bot
"""
import os
import json
import requests
from flask import Flask, request, abort
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import (MessageEvent, TextMessage, TextSendMessage,
                            FlexSendMessage, ImageMessage)
from imgur_python import Imgur
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

try:
    with open('/home/config.json', 'r') as f:
        CONFIG = json.load(f)
    f.close()

    SUBSCRIPTION_KEY = CONFIG['azure']['subscription_key']
    ENDPOINT = CONFIG['azure']['endpoint']

    FACE_KEY = CONFIG['azure']['face_key']
    FACE_END = CONFIG['azure']['face_end']

    LINE_SECRET = CONFIG['line']['line_secret']
    LINE_TOKEN = CONFIG['line']['line_token']

    IMGUR_CONFIG = CONFIG['imgur']

except FileNotFoundError:
    SUBSCRIPTION_KEY = os.getenv('SUBSCRIPTION_KEY')
    ENDPOINT = os.getenv('ENDPOINT')
    FACE_KEY = os.getenv('FACE_KEY')
    FACE_END = os.getenv('FACE_END')
    LINE_SECRET = os.getenv('LINE_SECRET')
    LINE_TOKEN = os.getenv('LINE_TOKEN')
    IMGUR_CONFIG = {
        "client_id": os.getenv('IMGUR_ID'),
        "client_secret": os.getenv('IMGUR_SECRET'),
        "access_token": os.getenv('IMGUR_ACCESS'),
        "refresh_token": os.getenv('IMGUR_REFRESH')
    }

CV_CLIENT = ComputerVisionClient(
    ENDPOINT, CognitiveServicesCredentials(SUBSCRIPTION_KEY))
LINE_BOT = LineBotApi(LINE_TOKEN)
HANDLER = WebhookHandler(LINE_SECRET)
IMGUR_CLIENT = Imgur(config=IMGUR_CONFIG)


def azure_describe(url):
    """
    Output azure image description result
    """
    description_results = CV_CLIENT.describe_image(url)
    output = ""
    for caption in description_results.captions:
        output += "'{}' with confidence {:.2f}% \n".format(
            caption.text, caption.confidence * 100)
    return output


class AzureImageOutput():
    def __init__(self, url, filename):
        self.url = url
        self.filename = filename
        self.img = Image.open(filename)
        self.draw = ImageDraw.Draw(self.img)
        self.fnt = ImageFont.truetype(
            "static/TaipeiSansTCBeta-Regular.ttf",
            size=int(5e-2 * self.img.size[1]))

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
                print("{} at location {}, {}, {}, {}".format(
                    name, left, right, top, bot))
                self.draw.rectangle(
                    [left, top, right, bot], outline=(255, 0, 0), width=3)
                self.draw.text(
                    [left, abs(top - 12)],
                    "{} {}".format(name, confidence),
                    fill=(255, 0, 0),
                    font=self.fnt)

    def azure_face_detection(self):
        face_api_url = '{}face/v1.0/detect'.format(FACE_END)
        headers = {'Ocp-Apim-Subscription-Key': FACE_KEY}
        params = {
            'returnFaceId': 'true',
            'returnFaceLandmarks': 'false',
            'returnFaceAttributes': 'emotion',
        }
        response = requests.post(
            face_api_url,
            params=params,
            headers=headers,
            json={"url": self.url})
        if len(response.json()) > 0:
            for obj in response.json():
                left = obj['faceRectangle']['left']
                top = obj['faceRectangle']['top']
                right = obj['faceRectangle']['left'] + obj['faceRectangle']['width']
                bot = obj['faceRectangle']['top'] + obj['faceRectangle']['height']
                emotion = max(
                    obj["faceAttributes"]['emotion'],
                    key=obj["faceAttributes"]['emotion'].get)
                confidence = max(obj["faceAttributes"]['emotion'].values())
                self.draw.rectangle(
                    [left, top, right, bot], outline=(255, 0, 0), width=3)
                self.draw.text(
                    [left, abs(top - 12)],
                    "{} {}".format(emotion, confidence),
                    fill=(255, 0, 0),
                    font=self.fnt)

    def __call__(self):
        self.azure_object_detection()
        self.azure_face_detection()
        self.img.save(self.filename)
        image = IMGUR_CLIENT.image_upload(self.filename, 'first', 'first')
        link = image['response']['data']['link']
        os.remove(self.filename)
        return link


@app.route("/")
def hello():
    "hello world"
    return "Hello World!!!!!"


@app.route("/callback", methods=['POST'])
def callback():
    """
    LINE bot webhook callback
    """
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    print(body)
    try:
        HANDLER.handle(body, signature)
    except InvalidSignatureError:
        print(
            "Invalid signature. Please check your channel access token/channel secret."
        )
        abort(400)
    return 'OK'


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
        with open(filename, 'wb') as f_w:
            for chunk in message_content.iter_content():
                f_w.write(chunk)
        f_w.close()
        image = IMGUR_CLIENT.image_upload(filename, 'first', 'first')
        link = image['response']['data']['link']
        output = azure_describe(link)
        az_output = AzureImageOutput(link, filename)
        link = az_output()
        with open('templates/detect_result.json', 'r') as f_r:
            bubble = json.load(f_r)
        f_r.close()
        bubble['body']['contents'][0]['contents'][0]['contents'][0][
            'text'] = output
        bubble['header']['contents'][0]['contents'][0]['contents'][0][
            'url'] = link
        LINE_BOT.reply_message(
            event.reply_token,
            [FlexSendMessage(alt_text="Report", contents=bubble)])
