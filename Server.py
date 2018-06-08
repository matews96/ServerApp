import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
import joblib
import cv2
import numpy as np

clustering = joblib.load("/Users/mateo.echeverri/PycharmProjects/untitled3/models/kmeans_500_64.pkl")
classification = joblib.load("/Users/mateo.echeverri/PycharmProjects/untitled3/models/svm_lineal_500.pkl")
surf = cv2.xfeatures2d.SURF_create()

UPLOAD_FOLDER = '/tmp/'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def start():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return classify(os.path.join(app.config['UPLOAD_FOLDER'], filename))


def classify(filePath):
    img = cv2.imread(filePath, 0)
    kp, descriptors = surf.detectAndCompute(img, None)
    img = clustering.predict(descriptors)
    hist = np.histogram(img, bins=np.array(range(500)), density=True)
    os.remove(filePath)
    return str(classification.predict([hist[0]]))

if __name__ == '__main__':
    app.run()
