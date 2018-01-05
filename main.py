from nn import Classifier

from flask import Flask, render_template, request
from flask_cors import CORS
import base64
import os
import numpy as np

app = Flask(__name__)
classifier = Classifier()
CORS(app, headers=['Content-Type'])


@app.route("/", methods=["POST", "GET", 'OPTIONS'])
def index_page():
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')


@app.route('/hook', methods=["GET", "POST", 'OPTIONS'])
def predict():
    answer = ""

    if request.method == 'POST':
        image_b64 = request.values['imageBase64']
        image_encoded = image_b64.split(',')[1]
        image = base64.decodebytes(image_encoded.encode('utf-8'))
        prediction = classifier.predict(image)
        if prediction == "Draw something please :)":
            return prediction
        arr = prediction['cnn_a'][0]
        answer = str(np.argmax(arr)+1) + ' (' + str(max(arr)*100) + '%)'
    return answer

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 6025))
    app.run(host='0.0.0.0', port=port, debug=False)
