import io
from PIL import Image
import cv2
import torch
from flask import Flask, render_template, request, make_response, Response
from flask_cors import CORS, cross_origin
from werkzeug.exceptions import BadRequest
import os
import sys
import json

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
dictOfModels = {}

# create a list of keys to use them in the select part of the html code
listOfKeys = []


def get_prediction(img_bytes, model):
    img = Image.open(io.BytesIO(img_bytes))
    results = model(img, size=640)
    return results

# Route
# GET


@app.route('/', methods=['GET'])
def get():
    # in the select we will have each key of the list in option
    return render_template("index.html", len=len(listOfKeys), listOfKeys=listOfKeys)
# POST


@app.route('/', methods=['POST'])
@cross_origin()
def predict():
    file = extract_img(request)
    img_bytes = file.read()
    # Choice of the model
    results = get_prediction(
        img_bytes, dictOfModels[request.form.get("model_choice")])
    # app.logger(f'User selected model : {request.form.get("model_choice")}')
    # Updates Result images with boxes and labels
    results.render()
    # Encoding the resulting image and return it

    if request.form.get('result_type') == 'json':
        temp = results.pandas().xyxy[0].to_json(orient="records")
        parsed = json.loads(temp)
        response = json.dumps(parsed, indent=4)
    else:
        for img in results.imgs:
            RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im_arr = cv2.imencode('.jpg', RGB_img)[1]
            response = make_response(im_arr.tobytes())
            response.headers['Content-Type'] = 'image/jpeg'

    # Return image with boxes and labels or json
    return Response(response, mimetype='application/json')


def extract_img(request):
    # checking if image uploaded is valid
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")
    file = request.files['file']
    if file.filename == '':
        raise BadRequest("Given file is invalid")
    return file


# Entry
if __name__ == '__main__':
    print('Starting yolov5 webservice...')
    # Getting directory containing models from command args (or default 'models_train')
    models_directory = 'models_train'
    if len(sys.argv) > 1:
        models_directory = sys.argv[1]
    print(f'Watching for yolov5 models under {models_directory}...')
    for r, d, f in os.walk(models_directory):
        for file in f:
            if ".pt" in file:
                # example: file = "model1.pt"
                # the path of each model: os.path.join(r, file)
                model_name = os.path.splitext(file)[0]
                model_path = os.path.join(r, file)
                print(
                    f'Loading model {model_path} with path {model_path}...')
                dictOfModels[model_name] = torch.hub.load(
                    'ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
                # you would obtain: dictOfModels = {"model1" : model1 , etc}
        for key in dictOfModels:
            listOfKeys.append(key)  # put all the keys in the listOfKeys

    print(
        f'Server now running on ')

    # starting app
    app.run(debug=True, host='0.0.0.0')
