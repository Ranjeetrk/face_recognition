import face_recognition
from PIL import Image


def get_face_locations():
    image = face_recognition.load_image_file("images/pic.jpg")
    face_locations = face_recognition.face_locations(image)
    i = 0
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save("face_detected/face-{}.jpg".format(i))
        i = i + 1
