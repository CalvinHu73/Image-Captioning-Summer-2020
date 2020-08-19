import os
import time
import hashlib
import json

import urllib.request
from flask import Flask, render_template, redirect, url_for, request, flash
from werkzeug.utils import secure_filename

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'static/uploads') # you'll need to create a folder named uploads
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.secret_key = "test"
def allowed_image(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":

        if request.files:
            if 'image' not in request.files:
                flash('No image uploaded!')
                return redirect(request.url)

            image = request.files["image"]
           
            if image.filename == "":
                print("No filename")
                flash('No image selected for uploading')
                return redirect(request.url)

            if allowed_image(image.filename):
                filename = secure_filename(image.filename)

                image.save(os.path.join(app.config["UPLOADED_PHOTOS_DEST"], filename))
                flash('Image successfully uploaded and displayed')
                print("Image saved")
                print(filename)
                return render_template("upload.html", filename=filename)

            else:
                flash('Allowed image types are -> png, jpg, jpeg, gif')
                print("That file extension is not allowed")
                return redirect(request.url)

    return render_template("upload.html")

@app.route('/display/<filename>')
def display_image(filename):
    print("Running display")
	#print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

from pprint import pprint

@app.route('/examples')
def examples():
    with open("examples.json", "r") as examplesFile:    
        exampleArray = json.load(examplesFile)
    print(exampleArray[0]["img"])
    return render_template("examples.html", exampleArray=exampleArray)


if __name__ == "__main__":
  app.run(debug=True)