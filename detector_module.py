import cv2
import os


working_dir = os.getcwd()
cascades_dir = os.path.join(working_dir, "cascades")

CASCADES = {
    "eye": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_eye.xml")),
    "smile": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_smile.xml")),
    "cat_face": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_frontalcatface_extended.xml")),
    "face": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_frontalface_default.xml")),
    "face2": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_frontalface_alt2.xml")),
    "full_body": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_fullbody.xml")),
    "lower_body": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_lowerbody.xml")),
    "upper_body": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_lowerbody.xml")),
    "russian_number_plate": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_licence_plate_rus_16stages.xml"))
}


def detect_face_elements(gray, frame, to_detect):
    faces = CASCADES["face"].detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_face_gray = gray[y:y+h, x:x+w]
        roi_face_color = frame[y:y+h, x:x+w]
        if "eye" in to_detect:
            eyes = CASCADES["eye"].detectMultiScale(roi_face_gray, 1.1, 22)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_face_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        if "smile" in to_detect:
            smiles = CASCADES["smile"].detectMultiScale(roi_face_gray, 1.7, 22)
            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_face_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    return frame




