import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from keras_preprocessing.image import img_to_array

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from pprint import pprint
import pathlib

#TODO : LORS Du premier entraînement sans augmentation de données ni introduction de dropout dans le réseau :
# - Comme vous pouvez le voir sur les graphiques (data_after_execution/execution_wthout_aug/training_data.png,
# la précision de l'entraînement et la précision de la validation sont largement faussées et le modèle n'a atteint
# qu'une précision d'environ 60% sur l'ensemble de validation.
# - Nous avons Regardé ce qui n'allait pas et nous avons essayé d'augmenter les performances globales du modèle en utilisant :
# l'augmentation de données et l'introduction de dropout dans le réseau
# Regarder le fichier Surrapprentisage.txt pour en savoir plus
from tensorflow.python.keras.applications.densenet import preprocess_input


def show_img(path, name):
    """cv2.imshow(name,img)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image"""
    plt.figure(figsize=(10,10))
    image = cv2.imread(path)
    plt.imshow(image)
    plt.suptitle(name)
    plt.axis("off")
    plt.show()

def visualisation_data(mTrain_ds, names_class):
    plt.figure(figsize=(10, 10))
    for images, labels in mTrain_ds.take(1):
      for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(names_class[labels[i]])
        plt.axis("off")
    plt.show()

def visualisation_datas_augmented(data_aug, train_ds):
    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_aug(images)
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
    plt.show()

def showBasedatas(pathWithMask, pathWithoutMask, pathMaskWearedMask):
    print("")
    print("")
    print("*********************************************************************************************************")
    print("*                                                                                                       *")
    print("*                         IA DETECTION DU PORT DE MASQUE                                                *")
    print("*                                                                                                       *")
    print("*********************************************************************************************************")
    print()
    print()

#TODO : -------------------------------------------- 1 - PREPARATION DE NOTRE DATASET --------------------------------------------------------------------------------------------
def BuildDataset(current_folder):
    print()
    print()
    print("    *****************************************************************")
    print("    *           PREPARATION DE NOTRE DATASET                        *")
    print("    *****************************************************************")
    print()
    # Nous utilisons un ensemble de données utilisant beaucoup d'images.
    # L'ensemble de données contient 03 sous répertoires, un par classe:
    # todo : 'mask_weared_incorrect, 'without_mask' et with_mask'

    # CONSTRUCTION DE NOTRE DATASET, CECI EST POUR RECUPERER UN DATSAET EN LIGNE et/OU en local:
    # dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    # data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    # data_dir = pathlib.Path(data_dir)
    # print(data_dir)

    dataset_dir = pathlib.Path(os.path.join(current_folder, r'dataset'))  # répertoire du dataset
    image_count = len(list(dataset_dir.glob('*/*.jpg')))  # nombre d'image que contient notre dataset
    print(f" - Nombre total d'image dans notre dataset : {image_count}")
    print(f" - Location de notre dataset : {dataset_dir}")
    print(" - Trois images vous sont affchées pour être sûr de notre dataset")
    incorrect_mask = list(dataset_dir.glob('mask_weared_incorrect/*'))  # On affiche quelques images de  masques portés mals
    good_mask = list(dataset_dir.glob('with_mask/*'))  # On affiche une image avec masque
    no_mask = list(dataset_dir.glob('without_mask/*'))  # On affche une image sans masque
    # un masque mal porté
    show_img(str(incorrect_mask[16]), 'UN MASQUE MAL PORTE')
    show_img(str(good_mask[32]), 'AVEC MASQUE')
    show_img(str(no_mask[20]), 'SANS MASQUE')

    #todo : On appelle la méthode de création du jeu de donnée
    creationJeuDeDonne(dataset_dir)

#TODO : -------------------------------------------- 2 - CREATION DE NOTRE JEU DE DONNEE --------------------------------------------------------------------------------------------
def creationJeuDeDonne(dataset_dir):
    print()
    print()
    print("    *****************************************************************")
    print("    *           2 - CREATION DE NOTRE JEU DE DONNEE                 *")
    print("    *****************************************************************")
    print()

    # Définissons quelques paramètres pour le chargeur:
    batch_size = 32
    img_height = 224
    img_width = 224
    print(f" -Nos données : ")
    print(f"     *batch size :{batch_size} ")
    print(f"     *Image Height : {img_height} ")
    print(f"     *Image Wdth : {img_height} ")
    print()

    # Il est recommandé d'utiliser une séparation de validation lors du développement de notre modèle.
    # Utilisons 80% des images pour la formation et 20% pour la validation.
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(  # Nos paramètres d'entrainement
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(  # Nos paramètres de validation
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    # On peur récuperer les noms des classes dans l'attribut "class_names" sur nos ensembles de données.
    # Ils correspondent aux noms des répertoires par ordre alphabétique
    class_names = train_ds.class_names
    print(f"Suivant les données d'entraînement, nos classes sont : {class_names}")

    #todo : On appel la visualisation des données
    dataVisualisation(train_ds, class_names)
    #Todo : augmentation des données
    # Voir Surrapprentissage.txt
    data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal",
                                                         input_shape=(img_height,
                                                                      img_width,
                                                                      3)),
            layers.experimental.preprocessing.RandomRotation(0.1),
            layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )
    #todo : Visualisation des données augmentées
    visualisation_datas_augmented(data_augmentation, train_ds)
    #todo: Appel configuration des données pour les performances
    configDataAccuration(train_ds, val_ds)
    #todo : Appel créaton du model
    createModel(data_augmentation, train_ds, val_ds)


#TODO : ----------------------------------- 3 - VISUALISATION LES DONNEES DE NOTRE DATASET  --------------------------------------------------------------------------------------------
def dataVisualisation(train_ds, class_names ):
    # Nous allons visualiser les 9 premières images de notre train_data
    visualisation_data(train_ds, class_names)
    # Nous allons entraîner un modèle à l'aide de ces ensembles de données en les passant à model.fit dans un instant.
    # mais nous souhaitons d'abord, parcourir manuellement l'ensemble de données et récupérer des lots d'images:
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break



#TODO : --------------------- 5 - CONFIGURATION DE L'ENSEMBLE DES DONNEES POUR LES PERFORMANCES --------------------------------------------------------------------------------------------
def configDataAccuration(train_ds, val_ds):

    # Veillons à utiliser la prélecture tamponnée afin de pouvoir générer des données à partir du disque sans que les
    # E / S deviennent bloquantes. Nous allons utiliser deux méthodes précieuses lors du chargement de données.
    # Dataset.cache() garde les images en mémoire après leur chargement hors du disque au cours de la première époque.
    # Dataset.prefetch() chevauche le prétraitement des données et l'exécution du modèle pendant l'entraînement.
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#TODO : ------------------- 6 - CREATION DU MODEL (EN APPLQUANT LE DROPOUT et le data_augmentation pour un Surraprentissage) --------------------------------------------------------------------------------------------
def createModel (data_augmentation, train_ds, val_ds):
    # Le modèle se compose de trois blocs de convolution avec une couche de pool maximale dans chacun d'eux.
    # Il y a une couche entièrement connectée avec 128 unités au-dessus qui est activée par une fonction d'activation relu
    # Ce modèle n'a pas été réglé pour une grande précision, nous allons avoir une approche standard.

    num_classes = 3

    model = Sequential([
        data_augmentation,  # Notre augentation de données
        layers.experimental.preprocessing.Rescaling(1. / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),  # DropOut
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    #todo: Appel compilation model
    compilModel(model)
    #todo: Appel résumé du model
    showSummaryModel(model)
    #todo: Appel train model
    TrainModel(model, train_ds, val_ds)


#TODO : --------------------------------------- 7 - COMPILATION DU MODEL --------------------------------------------------------------------------------------------
def compilModel(model):
    # choisissons les optimizers.Adam optimizer and losses.SparseCategoricalCrossentropy loss function.
    # Pour afficher la précision de l'entraînement et de la validation pour chaque époque d'entraînement,
    # Nous allons transmettre l'argument metrics .
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])


#TODO : ---------------------------------------- 8 -  RESUME DU MODEL --------------------------------------------------------------------------------------------
def showSummaryModel(model):
    # Visualisons toutes les couches du réseau en utilisant la méthode summary du modèle:
    model.summary()


#TODO : -------------------------------------- 9 - : FORMATION DU MODEL --------------------------------------------------------------------------------------------
def TrainModel(model, train_ds, val_ds):
    epochs = 32
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    # Sauvegarde du modèle
    model.save("model_trainFinal_30.model", save_format="h5")

    #todo: Appel de l'historique
    trainingHISTORY(history, epochs)

#TODO : ------------------------------ 10 - VISUALISATION DE LA PRECISION D'ENTRAINEMENT ET DE VALIDATION --------------------------------------------------------------------------------------------
def trainingHISTORY(history, epochs):
    # Nous allons Créer des graphiques de perte et de précision sur les ensembles de formation et de validation.
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


#TODO : ------------------------------  11 -  PREDICTION DU MODEL --------------------------------------------------------------------------------------------
def predict_image(modelMask, current_folder, img_height, img_width, class_names,
                  labels_dict, color_dict, modelCnn, prototxt_path, caffemodel_path ):

    print("veuillez entrer le nom de l'image + son extension : ")
    # load our serialized face detector model from disk
    print("[INFO] loading face detector model...")
    net = cv2.dnn.readNet(prototxt_path, caffemodel_path)
    image_input = os.path.join(current_folder, "Images_pour_test/bb_test.jpg")
    # load the input image from disk, clone it, and grab the image spatial
    # dimensions
    image = cv2.imread(image_input)
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            img = keras.preprocessing.image.load_img(
                image_input, target_size=(224, 224)
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)

            predictions = modelMask.predict(img_array)
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
            path_img = os.path.join(os.getcwd(), 'resultats_predict/result_predict5.png')

            if classe_found == "mask_weared_incorrect":
                showstr = labels_dict[0] + ": " + scoreStr
                cv2.rectangle(image, (startX, startY), (endX, endY), color_dict[0], 2)
                cv2.rectangle(image, (startX, startY - 40), (endX, startY), color_dict[0], -1)
                result = cv2.putText(image, showstr, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow('result', result)
                cv2.imwrite(path_img, result)
                show_img(path_img, "RESULTAT")

            elif classe_found == "with_mask":
                showstr = labels_dict[1] + ": " + scoreStr
                cv2.rectangle(image, (startX, startY), (endX, endY), color_dict[1], 2)
                cv2.rectangle(image, (startX, startY - 40), (endX, startY), color_dict[1], -1)
                result = cv2.putText(image, showstr, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow('result', result)
                cv2.imwrite(path_img, result)
                show_img(path_img, "RESULTAT")
            elif classe_found == "without_mask":
                showstr = labels_dict[2] + ": " + scoreStr
                cv2.rectangle(image, (startX, startY), (endX, endY), color_dict[2], 2)
                cv2.rectangle(image, (startX, startY - 40), (endX, startY), color_dict[2], -1)
                result = cv2.putText(image, showstr, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow('result', result)
                cv2.imwrite(path_img, result)
                show_img(path_img, "RESULTAT")


def predictVideo():
    import live_detection




#TODO: --------------------------------------------------------------------------------------------------------------------------------------------------------------



#TODO --------------------------------------------- PROGRAMME PRINCIPAL ---------------------------------------------------------------------------------

# Detection de visages à l'aide du model Cafee Model Zoo
# http://caffe.berkeleyvision.org/model_zoo.html
prototxt_path = os.path.join('caffe_parametres/deploy.prototxt')
caffemodel_path = os.path.join('caffe_parametres/weights.caffemodel')
model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
# Chargement du modèle permettant de détecter le port du masque
modelMasque = tf.keras.models.load_model("model_trainFinal_25.model")
#On définit les etiquettes qui seront affichées
labels_dict = {0:'Mask Wear incorrect', 1:'With Mask', 2:'Without Mask'}
class_names = ['mask_weared_incorrect', 'with_mask', 'without_mask']
#On définit les couleurs pour chaque sorties détectées (en BGR colors)
color_dict = {0:(0, 255, 255), 1:(0,255,0), 2:(0,0,255)}#0: jaune, 1: vert, 2: rouge
img_height = 224
img_width = 224
current_folder = os.getcwd()  # répertoire courant
#todo : Construction de notre dataset : elle va appeler toutes les autres méthodes
#BuildDataset(current_folder)
modelPath = os.path.join(current_folder, r'model_trainFinal_25.model')

print("")
print("")
print("*********************************************************************************************************")
print("*                                                                                                       *")
print("*                                         MENU PRINCIPAL                                                *")
print("*                                                                                                       *")
print("*********************************************************************************************************")
print()
print()

print(" - Pour prédire une image statique entrez 1 :")
print(" - Pour prédire dans un flux vidéo en temps réel entrez 2 :")
print(" - Pour entraîner un model avec le dataset collectif du projet (local) entrez 3 :")

choix = str(input())

if choix == '1':
    predict_image(modelMasque,current_folder,img_height,img_width, class_names,
                  labels_dict, color_dict, model,prototxt_path, caffemodel_path)
    pass
elif choix == '2':
    predictVideo()
    pass
elif choix == '3':
    BuildDataset(current_folder)
    pass


"""if not os.path.exists(modelPath):
    BuildDataset(current_folder)
else:
    # Chargement du modèle permettant de détecter le port du masque
    class_name = ['mask_weared_incorrect', 'with_mask', 'without_mask']
    modelMasque = tf.keras.models.load_model("model_trainFinal.model")
    predictModel(modelMasque, current_folder, img_height, img_width, class_name)"""


