import cv2
import os


def get_face_locations(image):
    detector = cv2.CascadeClassifier('venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    # image = cv2.imread('images/pic.jpg')
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,1.3,5)
    base = os.path.abspath(os.path.dirname(__file__))
    base = os.path.join(base, "face_detected")
    i = 0
<<<<<<< HEAD
    for(x,y,w,h) in faces:
        face_image = image[y:y+h,x:x+w]
        temp = os.path.join(base, "face-"+str(i))
        cv2.imwrite(temp + '.jpg', face_image)
        i = i + 1
=======
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save("face_detected/face-{}.jpg".format(i))
        i = i + 1
  

>>>>>>> d715cc6e81537b6bc31392906def4a474f0b6bde
