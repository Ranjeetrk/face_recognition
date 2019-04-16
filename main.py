import os
import pickle
from keras import Model
from capture import take_image
from face_location import get_face_locations
from vgg_net import preprocess_image, findEuclideanDistance
from keras.applications.vgg16 import VGG16
# load the model
model = VGG16()
vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
epsilon = 45

path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(path, "face_vectors")
dirs = os.listdir(path)

print('Do you want to verify? (y/n)')
condition = True if input() == 'y' else False

while condition:
    print('Press \'q\' to take picture')
    take_image()
    get_face_locations()
    cur_path = os.path.abspath(os.path.dirname(__file__))
    cur_path = os.path.join(cur_path, "face_detected")
    representations = []
    for face in os.listdir(cur_path):
        img_representation = vgg_face_descriptor.predict(preprocess_image(os.path.join(cur_path, face)))[0, :]
        representations.append(img_representation)
    for img in representations:
        for dirt in dirs:
            temp_path = os.path.join(path, dirt)
            file = os.listdir(temp_path)
            with open(os.path.join(temp_path, file[0]), 'rb') as f:
                img_rep = pickle.load(f)
            euclidean_distance = findEuclideanDistance(img, img_rep)
            if euclidean_distance < epsilon:
                print("Hello "+dirt+" !")
                print(euclidean_distance)
    print("Do you want to continue? (y/n)")
    condition = True if input() == 'y' else False
    
    
