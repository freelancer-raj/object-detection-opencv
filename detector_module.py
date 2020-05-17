import cv2
import os


working_dir = os.getcwd()
cascades_dir = os.path.join(working_dir, "cascades")

CASCADES = {
    "eye": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_eye.xml")),
    "smile": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_smile.xml")),
    "profile": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_profileface.xml")),
    "cat_face": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_frontalcatface_extended.xml")),
    "face": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_frontalface_default.xml")),
    "face2": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_frontalface_alt2.xml")),
    "full_body": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_fullbody.xml")),
    "lower_body": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_lowerbody.xml")),
    "upper_body": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_lowerbody.xml")),
    # "russian_number_plate": cv2.CascadeClassifier(os.path.join(cascades_dir, "haarcascade_licence_plate_rus_16stages.xml"))
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
        if "profile" in to_detect:
            profiles = CASCADES["profile"].detectMultiScale(roi_face_gray, 1.7, 22)
            for (px, py, pw, ph) in profiles:
                cv2.rectangle(frame, (px, py), (px+pw, py+ph), (255, 0, 0), 2)
    return frame


def detect_others(gray, frame, to_detect):
    objects = CASCADES[to_detect].detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return frame


# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # canvas = detect_face_elements(gray, frame, ["face", "eye", "smile", "profile"])
    canvas = detect_others(gray, frame, "cat_face")
    cv2.imshow('Video', canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()



