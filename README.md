# Face Recognition and Verification
Face Recognition and Verification are widely used for security and identification in recent years.
So using this model, users can recognize a person's face and verify the person from the stored database. This project developed for attendance system using face Recognition.
## VGG16
Using transfer-learning, I have take VGG16 model for extracting the facial features. Which will be used to compare to a person's face and verify by choosing threshold value. [code](https://github.com/Ranjeetrk/face_recognition/blob/master/vgg_net.py)
##
 - I have used OpenCV haarcascade_frontalface_default for getting face location coordinate from the input image.
 - Now we have to create a vector of all the faces using the transfer-learning concept, and these face vectors are to be compared with existing face_vectors stored in face_vectors directory.
 - For comparison of two vectors, I have used Euclidean distance between both vectors. If this distance is less than some threshold (e.g., epsilon=45), then we say they are the same person.
 - For creating dataset run [dataset_creater]([https://github.com/Ranjeetrk/face_recognition/blob/master/dataset_creater.py](https://github.com/Ranjeetrk/face_recognition/blob/master/dataset_creater.py))
 - Run [main.py]([https://github.com/Ranjeetrk/face_recognition/blob/master/main.py](https://github.com/Ranjeetrk/face_recognition/blob/master/main.py)) for face verification .
