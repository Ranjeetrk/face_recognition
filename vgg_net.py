from keras import Model
from keras.preprocessing import image
import numpy as np
from keras_vggface import utils
# model = Sequential()
# model.load_weights('vgg_face_weights.h5')

from keras.applications.vgg16 import VGG16
# load the model
model = VGG16()

epsilon = 120  # euclidean distance


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = utils.preprocess_input(img, version=1)
    return img


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)


def verifyFace(img1, img2):

    img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))[0,:]
    img2_representation = vgg_face_descriptor.predict(preprocess_image(img2))[0,:]
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)

    if euclidean_distance < epsilon:
        print("verified... they are same person")
    else:
        print("unverified! they are not same person!")


# verifyFace('images/rishi.png', 'images/rishi29.png')
