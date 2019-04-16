# face_recognition
Develop face_recognition model using transfer-learning .
I have used VGG16() model with pre-trained 
created main.py file for main operation of face_recognition ,At first call capture.py to take image from webcam and stored in folder Images
secondly we want all the face-laction from that Image for that we have created face_location.py which will be called after the capture.py in the main and stored all the detected faces in face_detected folder.
Now we have to creat vector of all the faces using transfer-learning concept.and these face vectors are to be compaired with existing face_vectors stored in face_vectors directory. This face_vectors directory contains all the face vectors of known people from our dataset.
For the purpose of comparison of two vectors we get the eucledian distance between both vectors and if this distance is less than some threshold, epsilon=45 in our case, then these are same persons. 
Check the main.py file for detailed code.

