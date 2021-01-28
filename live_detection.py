import cv2
import os
import numpy as np
import tensorflow as tf
from tensorboard.util import encoder
from tensorflow import int64
from keras.utils import np_utils, to_categorical
from pprint import pprint
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from playsound import playsound

def maximumIndices(liste):
    maxi = liste[0]
    longueur=len(liste)
    for i in range(longueur):
        if liste[i] >= maxi:
            maxi = liste[i]
    return maxi

# Detection de visages à l'aide du model Cafee Model Zoo
# http://caffe.berkeleyvision.org/model_zoo.html
prototxt_path = os.path.join('caffe_parametres/deploy.prototxt')
caffemodel_path = os.path.join('caffe_parametres/weights.caffemodel')
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Chargement du modèle permettant de détecter le port du masque
modelMasque = tf.keras.models.load_model("model_trainFinal_25.model")

# Capture de la caméra (idCamera)
cap = cv2.VideoCapture(0)

#On définit les etiquettes qui seront affichées
labels_dict = {0:'Mask Wear incorrect', 1:'With Mask', 2:'Without Mask'}
class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask']
#On définit les couleurs pour chaque sorties détectées (en BGR colors)
color_dict = {0:(0, 255, 255), 1:(0,255,0), 2:(0,0,255)}#0: jaune, 1: vert, 2: rouge

while True:

    _, image = cap.read()

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    model.setInput(blob)
    detections = model.forward()

    h = image.shape[0]
    w = image.shape[1]

    for i in range(0, detections.shape[1]):
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        confidence = detections[0, 0, i, 2]

        # If confidence > 0.5, save it as a separate file
        if (confidence > 0.5):
            frame = image[startY:endY, startX:endX]

            # Appel du modèle appris pour la detection de masque
            capture = cv2.resize(frame, (224, 224))
            path_img = os.path.join(os.getcwd(), 'resultats_predict/save_im.png')
            cv2.imwrite(path_img, capture)
            img = keras.preprocessing.image.load_img(
                path_img, target_size=(224, 224)
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch"""
        #reshaped = np.reshape(capture, (32, 224, 224, 3))
            predictions = modelMasque.predict(img_array)
            print(predictions)
            score = tf.nn.softmax(predictions[0])
            print(score)
            print(
                "This image most likely belongs to {} with a {:.2f} percent confidence."
                    .format(class_names[np.argmax(score)], 100 * np.max(score))
            )

            classe_found = str(format((class_names[np.argmax(score)])))
            score_real = 100 * np.max(score)
            scoreStr = str(score_real)

            if classe_found == "mask_weared_incorrect":
                            showstr = labels_dict[0] + ": " + scoreStr
                            cv2.rectangle(image, (startX, startY), (endX, endY), color_dict[0], 2)
                            cv2.rectangle(image, (startX, startY - 40), (endX, startY), color_dict[0], -1)
                            cv2.putText(image, showstr, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            elif classe_found == "with_mask":
                            showstr = labels_dict[1] + ": " + scoreStr
                            cv2.rectangle(image, (startX, startY), (endX, endY), color_dict[1], 2)
                            cv2.rectangle(image, (startX, startY - 40), (endX, startY), color_dict[1], -1)
                            cv2.putText(image, showstr, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            elif classe_found == "without_mask":
                            showstr = labels_dict[2] + ": " + scoreStr
                            cv2.rectangle(image, (startX, startY), (endX, endY), color_dict[2], 2)
                            cv2.rectangle(image, (startX, startY - 40), (endX, startY), color_dict[2], -1)
                            cv2.putText(image, showstr, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                            #path = os.path.abspath("Alarm.wav")
                            #playsound(path)

    # Affichage de l'image
    cv2.imshow('img', image)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break