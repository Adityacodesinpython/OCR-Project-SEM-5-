import os
from os import listdir
from os.path import isfile, join
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] ='vision.json'
from google.cloud import vision
from my_timer import my_timer
import time

def detect_text(path):
    client = vision.ImageAnnotatorClient()
    with open(path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    ocr_text = []
    for text in texts:
        ocr_text.append(f"\r\n{text.description}")
    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    # return texts[0].description
    return ocr_text

# @my_timersssss
def main():
    mypath = "mini-proj-assets/"
    only_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for image_path in only_files:
        text = detect_text(mypath+image_path)
        print(image_path)
        for i in text:
            try:
                float(i)
                print(i, end="")
            except ValueError:
                pass
            # print(i, end="")
        print("\n")

if __name__ == "__main__":
     main()
