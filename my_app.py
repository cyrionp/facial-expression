from flask import Flask, render_template, request, flash, redirect, url_for
from flask_caching import Cache
import numpy as np
import keras
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
import cv2
from mtcnn import MTCNN
from PIL import Image
import time

UPLOADS_DIR = "./static/uploads/"
FACES_DIR = "./static/faces/"
ALLOWED_FILES = {"png", "jpg", "jpeg"}
EXPRESSION_CATEGORIES = ("kızgın", "iğrenmiş", "korkmuş", "mutlu", "üzgün", "şaşırmış", "nötr")

config = {
    "DEBUG": False,
    "CACHE_TYPE": "SimpleCache",
    "CACHE_DEFAULT_TIMEOUT": 300,
    "MAX_CONTENT_LENGTH": 16 * 1024 * 1024
}

app = Flask(__name__)
app.secret_key = "Dusun53Ne1$01mu5ZLiR@cIK"
app.config.from_mapping(config)
cache = Cache(app)


def is_allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_FILES


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
    if file and is_allowed(file.filename):
        filename = secure_filename(file.filename)
        delete_files_in_dir(UPLOADS_DIR)
        file.save(os.path.join(UPLOADS_DIR, filename))
        flash("Resim yüklendi")
        flash(prediction(filename))
        return render_template("index.html", filename=filename)
    else:
        flash("İzin verilen resim uzantıları: png, jpg, jpeg")
        return redirect(request.url)


@app.route("/display/<filename>")
def display(filename):
    return redirect(url_for("static", filename="uploads/" + filename), code=301)


def delete_files_in_dir(directory):
    for i in os.listdir(directory):
        os.remove(directory + i)
        print(i + " silindi")


def verify_jpeg(path):
    try:
        img = Image.open(path)
        img.getdata()[0]
    except OSError:
        return False
    return True


@cache.cached(key_prefix="load_model_and_detector")
def load_model_and_detector():
    global model, detector
    start_timer = time.time()
    model = keras.models.load_model("mymodel.h5")
    print(f'içeride model load run süresi: {time.time() - start_timer}')
    detector = MTCNN()


def prediction_processes(filename):
    start_timer = time.time()
    load_model_and_detector()
    print(f'load_model_and_detector run süresi: {time.time() - start_timer}')

    img = cv2.imread(UPLOADS_DIR + filename)
    detections = detector.detect_faces(img)

    if detections:
        for face in detections:
            score = face["confidence"]
            if score >= 0.80:
                x, y, w, h = face["box"]
                detected_face = img[int(y):int(y + h), int(x):int(x + w)]

                delete_files_in_dir(FACES_DIR)
                cv2.imwrite(FACES_DIR + filename, detected_face)
                print("Resim kaydedildi: " + filename)

                img = image.load_img(FACES_DIR + filename, color_mode="grayscale", target_size=(48, 48))

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

                print('Yüz ifadeniz tahminennn: ', EXPRESSION_CATEGORIES[most_index])
                return f"Yüz ifadeniz tahminen: {EXPRESSION_CATEGORIES[most_index]}"
    else:
        return "Resimde yüz bulunamadı"


def prediction(filename):
    file_extension = os.path.splitext(UPLOADS_DIR + filename)[1].lower()

    if file_extension == ".jpeg":
        if verify_jpeg(UPLOADS_DIR + filename):
            return prediction_processes(filename)
    else:
        return prediction_processes(filename)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
