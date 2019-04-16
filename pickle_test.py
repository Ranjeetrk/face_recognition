from keras import Model
import pickle
import os
from vgg_net import preprocess_image, findEuclideanDistance
from keras.applications.vgg16 import VGG16
# load the model
model = VGG16()


vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)


def pickle_vector(img, name):

    img_representation = vgg_face_descriptor.predict(preprocess_image(img))[0, :]
    base = os.path.abspath(os.path.dirname(__file__))
    base = os.path.join(base, "face_vectors")
    base = os.path.join(base, name)
    with open(base+'.pkl', 'wb') as f:
        pickle.dump(img_representation, f)


def unpickle(file):

    with open(file, 'rb') as f:
        img_representation = pickle.load(f)
    return img_representation



pickle_vector('ranjeet.jpg', 'ranj')


"""base = os.path.abspath(os.path.dirname(__file__))
base = os.path.join(base, "face_vectors")

base1 = os.path.join(base, 'rk')
base2 = os.path.join(base, 'rk29')
img1_representation = unpickle(base1+'.pkl')
img2_representation = unpickle(base2+'.pkl')


euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)

if euclidean_distance < 45:
    print("verified... they are same person")
else:
    print("unverified! they are not same person!")"""

