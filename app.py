from ypd.pipeline.prediction import PredictionPipeline



if __name__ == "__main__":
    inference = PredictionPipeline()
    image_path = input('enter image path: ')
    pred = inference.predict(image_path)
    print(pred)