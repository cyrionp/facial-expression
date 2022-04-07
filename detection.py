from mtcnn import MTCNN
import cv2


def DetectFace(filename):
    detector = MTCNN()
    face_dir = "./static/faces/"

    img = cv2.imread(filename)
    detections = detector.detect_faces(img)

    for detection in detections:
        score = detection["confidence"]
        if score >= 0.90:
            x, y, w, h = detection["box"]
            detected_face = img[int(y):int(y+h), int(x):int(x+w)]
            cv2.imwrite(face_dir+filename, detected_face)
            print("Resim kaydedildi: "+filename)


DetectFace("IMG_1040.JPG")
