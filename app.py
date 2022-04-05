from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.preprocessing import image
import os
import urllib.request
from werkzeug.utils import secure_filename
from PIL import Image
import PIL
import time
import mediapipe
import cv2


uploads_folder = "./static/uploads"
allowed_files = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = "Dusun53Ne1$01mu5ZLiR@cIK"
app.config["UPLOAD_FOLDER"] = uploads_folder
app.config["MAX_CONTENT_LENGTH"] = 16*1024*1024


def isAllowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_files


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/", methods=["POST"])
def upload():
    if "file" not in request.files:
        flash("Dosya yok")
        return redirect(request.url)
    file = request.files["file"]
    if file.filename == "":
        flash("Yüklemek için bir resim seçmediniz")
        return redirect(request.url)
    if file and isAllowed(file.filename):
        filename = secure_filename(file.filename)
        _filename = "./static/uploads/"+filename

        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        #img = Image.open(_filename)
        #img.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        #resizeShowImage(filename)
        #resizePredictImage(filename)
        flash("Resim yüklendi")
        time.sleep(2)
        flash(Prediction(filename))
        return render_template("index.html", filename=filename)
    else:
        flash("İzin verilen resim uzantıları: png, jpg, jpeg")
        return redirect(request.url)


@app.route("/display/<filename>")
def display(filename):
    return redirect(url_for("static", filename="uploads/"+filename), code=301)


def PredictExpression(expressions):
    my_expressions = ('kızgın', 'iğrenmiş', 'korkmuş', 'mutlu', 'üzgün', 'şaşırmış', 'nötr')
    position = np.arange(len(my_expressions))
    plt.bar(position, expressions, align="center", alpha=0.9)
    plt.tick_params(axis="x", which="both", pad=10, width=4, length=10)
    plt.xticks(position, my_expressions)
    plt.ylabel("percentage")
    plt.title("expression")
    plt.show()


def Prediction(filename):
    #my_expressions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    my_expressions = ("kızgın", "iğrenmiş", "korkmuş", "mutlu", "üzgün", "şaşırmış", "nötr")
    model = keras.models.load_model("mymodel.h5")
    _filename = "./static/uploads/"+filename
    # Eğer göstermek için illa kaydetmemiz gerekiyorsa img.save deriz
    img = image.load_img(_filename, color_mode="grayscale", target_size=(48, 48))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255
    custom = model.predict(x)
    x = np.array(x, 'float32')
    x = x.reshape([48, 48])

    m = 0.000000000000000000001
    a = custom[0]
    for i in range(0, len(a)):
        if a[i] > m:
            m = a[i]
            ind = i

    print('Yüz ifadeniz tahminennn: ', my_expressions[ind])
    return f"Yüz ifadeniz tahminen: {my_expressions[ind]}"


if __name__ == "__main__":
    app.run(port=8080)
