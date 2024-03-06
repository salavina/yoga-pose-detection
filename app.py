from ypd.pipeline.prediction import PredictionPipeline
from ypd.utils.common import decodeImage
from flask_cors import CORS, cross_origin
import os
from flask import Flask, request, render_template, jsonify


os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


class YogaApp:
    def __init__(self):
        self.filename = "sample_image.jpg"
        self.classifier = PredictionPipeline()


@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@app.route("/train", methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    os.system("dvc repro")
    return "Training Done Successfully!"


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
   image = request.json['image']
   decodeImage(image, claApp.filename)
   prediction = claApp.classifier.predict(claApp.filename)
   return prediction

if __name__ == "__main__":
    claApp = YogaApp()
    
    
    app.run(host='0.0.0.0', port=8080) #for AWS & local host
    # app.run(host='0.0.0.0', port=80) #for Azure
    


# if __name__ == "__main__":
#     inference = PredictionPipeline()
#     image_path = input('enter image path: ')
#     pred = inference.predict(image_path)
#     print(pred)