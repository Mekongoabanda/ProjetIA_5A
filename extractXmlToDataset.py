import tensorflow as tf
from tensorflow import keras
import xmltodict
from lxml import etree
import numpy as np
import glob
import cv2, os
import labelme
from pprint import pprint
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#TODO: Pour comprendre les fichier XML avec python, visitez ce site : https://python.doctor/page-xml-python-xpath

def xml_to_dataset(dir, size=None): #dir : répertoire contenant les images ainsi que leurs Xml)
    tab_image = [] #Tableau d'images
    tab_label = [] #Tableau de nos labels
    counter = 0

    for fichier in glob.glob(dir + "/*.xml"): #Je veux récupérer tous les fichier .xml
        with open(fichier) as fd:#Je vais lire chaque fichier .xml rencontré
            doc = xmltodict.parse(fd.read())#Je stock le fichier .xml dans un tableau en gardant son arborescante
            image = doc['annotation']['filename'] #Je récupère le nom du fichier image correspondant au xml
            print(f"Image {counter}: {image}")
            counter+=1
            dir_img = os.path.join(dir, image)
            tree = etree.parse(fichier)#Variable pour parcourir notre fichier XML
            img = cv2.imread(dir_img)#On lit le fichier image
            for obj in  tree.xpath("/annotation/object") : #Je récupère le contenu de 'object' dans mon fichier .xml

                list_object = obj.getchildren()#liste des objets (name, pose, ..., bndbox)
                noeuds_name = list_object[0]# le noeuds 'name' est l'élément 0
                #le noeuds 4 (bnbox) comporte 04 autres enfants que nous cherchons
                noeuds_bndbox = list_object[4].getchildren()#liste des noeuds (xmin, xmax, ymin, ymax) du noeuds parent 'bndbox'
                #On récupère (xmin, xmax, ymin et ymax) et on les caste en int()
                xmin = int(float(noeuds_bndbox[0].text))
                ymin = int(float(noeuds_bndbox[1].text))
                xmax = int(float(noeuds_bndbox[2].text))
                ymax = int(float(noeuds_bndbox[3].text))

                if size is not None:#Si nous avons donné une taille en paramètre
                    tab_image.append(cv2.resize(img[ymin:ymax, xmin:xmax], size))#on ajoute l'image avec la taille requise
                else:#Sinon
                    tab_image.append(img[ymin:ymax, xmin:xmax])#On ajoute l'image telle qu'elle

                tab_label.append(noeuds_name.text)  # On ajoute le nom du label adéquat

    print(f"Nombre d'images : {len(tab_image)}")
    print(f"Nombre de labels : {len(tab_label)}")

    saveImageCrop(tab_image, tab_label)
    for i in range(10):
        show_img(tab_image[i], f"IMAGE {i+1} : {tab_label[i]} ")


#Classification des visages redimensionnés
def saveImageCrop(tab_image, tab_label):

    for i in range(len(tab_image)):
        print(f"Sauvegarde de l'image {i}...")
        if tab_label[i] == "with_mask":
            dir_image = os.path.join(r"dataset", tab_label[i]) #dossier "dataset/with_mask"
            if not os.path.exists(dir_image):  # Si le dossier "dataset/with_mask"  n'existe pas
                os.makedirs(dir_image)#On le crée
            name_image = os.path.join(dir_image, f"{tab_label[i]}_{i}.jpg" )#"dataset/with_mask/with_mask_[i].jpg"
            cv2.imwrite(name_image, tab_image[i])#On sauvegarde l'image

        elif tab_label[i] == "without_mask":
            dir_image = os.path.join(r"dataset", tab_label[i]) #dossier "dataset/without_mask"
            if not os.path.exists(dir_image):  # Si le dossier "dataset/without_mask"  n'existe pas
                os.makedirs(dir_image)#On le crée
            name_image = os.path.join(dir_image, f"{tab_label[i]}_{i}.jpg")#"dataset/without_mask/without_mask_[i].jpg"
            cv2.imwrite(name_image, tab_image[i])#On sauvegarde l'image

        else : #mask_weared_incorrect
            dir_image = os.path.join(r"dataset", tab_label[i]) #dossier "dataset/mask_weared_incorrect"
            if not os.path.exists(dir_image):  # Si le dossier "dataset/mask_weared_incorrect"  n'existe pas
                os.makedirs(dir_image)#On le crée
            name_image = os.path.join(dir_image, f"{tab_label[i]}_{i}.jpg")#"dataset/mask_weared_incorrect/mask_weared_incorrect_[i].jpg"
            cv2.imwrite(name_image, tab_image[i])#On sauvegarde l'image

def augmentedDataImage(myFiles):
    i = 0
    for file in myFiles:
        file_path = os.path.join(r"dataset/mask_weared_incorrect/", file )
        copyFile = cv2.imread(file_path)
        copyFileAUg = tf.image.flip_left_right(copyFile)
        plt.imshow(copyFile)
        plt.suptitle('img')
        plt.axis("off")
        plt.show()
        plt.imshow(copyFileAUg)
        plt.suptitle('AUG')
        plt.axis("off")
        plt.show()
        """print(copyFile)
        indice_fichier = i + 7616
        name_fichier = os.path.join(r"dataset/mask_weared_incorrect/mask_weared_incorrect_", str(indice_fichier) + ".jpg")
        cv2.imwrite(name_fichier, copyFileAUg)
        i += 1"""



def show_img(img, name):
    """cv2.imshow(name,img)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()  # destroys the window showing image"""
    plt.imshow(img)
    plt.suptitle(name)
    plt.axis("off")
    plt.show()

myDir = 'images'
xml_to_dataset(myDir, (224, 224))

myFilesMaskWeared = os.listdir('dataset/mask_weared_incorrect') #Liste des fichiers du dossier

data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1)#Tourne de manière aléatoire de 0.1
        ]
    )

#augmentedDataImage(myFilesMaskWeared)