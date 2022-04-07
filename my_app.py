from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
import keras
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
import time
import cv2
from mtcnn import MTCNN
from PIL import Image

uploads_dir = "./static/uploads/"
faces_dir = "./static/faces/"
allowed_files = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.secret_key = "Dusun53Ne1$01mu5ZLiR@cIK"
app.config["UPLOAD_FOLDER"] = uploads_dir
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


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
        DeleteFilesInDir(uploads_dir)
        file.save(os.path.join(uploads_dir, filename))
        flash("Resim yüklendi")
        time.sleep(2)
        flash(Prediction(filename))
        return render_template("index.html", filename=filename)
    else:
        flash("İzin verilen resim uzantıları: png, jpg, jpeg")
        return redirect(request.url)


@app.route("/display/<filename>")
def display(filename):
    return redirect(url_for("static", filename="uploads/" + filename), code=301)


def DeleteFilesInDir(directory):
    for i in os.listdir(directory):
        os.remove(directory + i)
        print(i + " silindi")


def VerifyJpeg(path):
    try:
        img = Image.open(path)
        img.getdata()[0]
    except OSError:
        return False
    return True


def Prediction(filename):
    my_expressions = ("kızgın", "iğrenmiş", "korkmuş", "mutlu", "üzgün", "şaşırmış", "nötr")

    def PredictionProcesses():
        model = keras.models.load_model("mymodel.h5")
        detector = MTCNN()

        img = cv2.imread(uploads_dir + filename)
        detections = detector.detect_faces(img)

        if detections:
            for face in detections:
                score = face["confidence"]
                if score >= 0.80:
                    x, y, w, h = face["box"]
                    detected_face = img[int(y):int(y + h), int(x):int(x + w)]

                    DeleteFilesInDir(faces_dir)
                    cv2.imwrite(faces_dir + filename, detected_face)
                    print("Resim kaydedildi: " + filename)

                    img = image.load_img(faces_dir + filename, color_mode="grayscale", target_size=(48, 48))

                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x /= 255
                    custom = model.predict(x)

                    minimum_value = 0.000000000000000000001
                    predictions = custom[0]
                    for i in range(0, len(predictions)):
                        if predictions[i] > minimum_value:
                            minimum_value = predictions[i]
                            most_index = i

                    print('Yüz ifadeniz tahminennn: ', my_expressions[most_index])
                    return f"Yüz ifadeniz tahminen: {my_expressions[most_index]}"
        else:
            return "Resimde yüz bulunamadı"

    file_extension = os.path.splitext(uploads_dir + filename)[1].lower()

    if file_extension == ".jpeg":
        if VerifyJpeg(uploads_dir + filename):
            return PredictionProcesses()
    else:
        return PredictionProcesses()


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
