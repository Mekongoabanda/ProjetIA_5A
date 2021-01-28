# PROJET DE 5A , SYSTEMES INTELLIGENTS AVANCES

![enter image description here](https://esirem.u-bourgogne.fr/wp-content/uploads/2019/12/cropped-logo-couleur-site-web.png)

## Auteurs:

 - **MEKONGO ABANDA Yannick - IT5 ILC**
 - **TONFE TCHATAT DAPHNIE - IT5 ILC**
 ## PRESENTATION DU PROJET :
 NB : Pour tester l'agorithme, vous aurez besoin de récupérer le modèle enregistré en cliquant sur --> : **[model_with_25_epoch.model](https://drive.google.com/file/d/1mTzKoHMU-8UYbKVg_aKmxhHGE449S1Hd/view?usp=sharing)**
 
 Ce projet avait pour objectif principal de nous familiariser avec les concepts d’apprentissage automatique, dans le domaine de l’intelligence artificiel. Il était question pour nous de concevoir un algorithme de détection du port de masque. Cet algorithme devait donc être capable de détecter si un individu porte bien son masque, porte mal ou ne porte pas de masque. Avant de présenter notre travail, il est important de présenter les outils/Framework que l’on a utilisé pour arriver à la solution finale.
 
Les trois scripts python utilisés sont :
 - **[classification.py](https://github.com/Mekongoabanda/ProjetIA_5A/blob/main/classificationTraining.py)** : Notre script à exécuter pour le lancement de l'algorithme. c'est dans ce script que s'effectue notre entraînement
 - **[live_detection.py](https://github.com/Mekongoabanda/ProjetIA_5A/blob/main/live_detection.py)** : C'est via ce script que se lance la prédiction temps réel .Il se lance à partir du fichier ci dessus.
 - **[(extractXmlToDataset.py)](https://github.com/Mekongoabanda/ProjetIA_5A/blob/main/extractXmlToDataset.py)** : qui permet de classifier notre Dataset en parcourant les annotations des images et en sauvegardant chaque visage dans son dossier correspondant.

 

### FRAMEWORK ET OUTILS UTILISES

 - Pour la préparation et l’entraînement de notre modèle nous avons utilisé KERAS + TensorFlow : ![KERAS ET TENSORFLOW](https://miro.medium.com/max/2560/0*BrC7o-KTt54z948C.jpg)
 - Pour la détection des visages nous avons utilisés CaffeModell via openCV :![enter image description here](https://user-images.githubusercontent.com/21311442/33640664-cbcbeff2-da6c-11e7-97c8-1ad8d7fdf4c0.png)

### La récupération automatique des visages pour une classification des données:

Nous avons opté pour une classification de nos données. A la base nous avions un dossier image contenant des images et leur annotations respectives en .xml. Ainsi nous avons écrit un script python **[(extractXmlToDataset.py)](https://github.com/Mekongoabanda/ProjetIA_5A/blob/main/extractXmlToDataset.py)** qui parcours les annotations (.XML), récupère les données de visages associées (et leur label) à l’image correspondante à celle-ci, puis sauvegarde chaque visage en tant qu’image avec pour nom son label + un numéro d’itération. Ainsi on a des données moins volumineuses dans notre espace de stockage. A la fin nous avons donc 03 sous-dossiers (mask_weared_incorrect, with_mask et without_mask) contenant les visages correspondants. A noter que les visages ont pour taille **224x224**

### La préparation de notre ensemble de données : 

Dans un dossier nommé **« dataset »** nous avons 03 classes donc 03 dossiers :

- Mask_weared_incorrect

- With_mask

- Without_mask

### La création de notre jeu de donnée :

Nous avons défini quelques paramètres pour le chargeur :

· **Batch_size = 32**

· Dimensions de l’image = **224x224**

· Nous avons utilisé une séparation de validation lors du développement de notre modèle.

· Nous utilisons **80% des images pour la formation** et **20% pour la validation**.
![enter image description here](https://github.com/Mekongoabanda/ProjetIA_5A/blob/main/data_after_execution/Capture.PNG?raw=true)

        train_ds =tf.keras.preprocessing.image_dataset_from_directory( # Nos paramètres d'entrainement
    
    dataset_dir,
    
    validation_split=0.2,
    
    subset="training",
    
    seed=123,
    
    image_size=(img_height, img_width),
    
    batch_size=batch_size)
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory( # Nos paramètres de validation
    
    dataset_dir,
    
    validation_split=0.2,
    
    subset="validation",
    
    seed=123,
    
    image_size=(img_height, img_width),
    
    batch_size=batch_size)

### VISUALISATION DE NOS DONNEES :
  
Nous générons un petit échantillon de nos données d’entraînement pour vérifier que chaque image est bien identifiée par son label :
![enter image description here](https://github.com/Mekongoabanda/ProjetIA_5A/blob/main/data_after_execution/Execution_without_aug/myplot.png?raw=true)

### CREATION DU MODEL, ENTRAINEMENT ET HISTORIQUE:
Après avoir eu plusieurs tentatives d’entrainement nous avons opté pour l’augmentation des données et le dropOut :

- **Statistiques sans augmentation des données et sans DropOut :** Regardez l’image ci-dessous, on remarque que notre modèle n’est pas tout à fait au point car la précision de l’entrainement et de la validation sont largement faussées.

![enter image description here](https://github.com/Mekongoabanda/ProjetIA_5A/blob/main/data_after_execution/Execution_without_aug/training_data.png?raw=true)

![enter image description here](https://github.com/Mekongoabanda/ProjetIA_5A/blob/main/data_after_execution/Execution_without_aug/training_with_10_epoch.PNG?raw=true)

- **Statistiques après augmentation des données et sans DropOut** Après avoir fait du surraprentissage à savoir l’augmentation des données et le dropOut on remarque que notre entraînement produit de bons résultats. Déjà l'augmentation des données se présente comme suit:

    data_augmentation = keras.Sequential( [layers.experimental.preprocessing.RandomFlip("horizontal", input_shape (img_height, img_width, 3)), layers.experimental.preprocessing.RandomRotation(0.1), layers.experimental.preprocessing.RandomZoom(0.1), ] )

![enter image description here](https://github.com/Mekongoabanda/ProjetIA_5A/blob/main/data_after_execution/Final_execution/myplot.png?raw=true)

On obtient un résulatat tout à fait correct : 
![enter image description here](https://github.com/Mekongoabanda/ProjetIA_5A/blob/main/data_after_execution/Final_execution/training_data_with_surraprentssage.png?raw=true)

# Prédiction de notre modèle :

Nous avons pu tester la prédiction de notre modèle en temps réel puis avec une image statique :

Voici quelques exemples de résultats :

## Prédiction par image statique :
![enter image description here](https://github.com/Mekongoabanda/ProjetIA_5A/blob/main/resultats_predict/result_predict1.png?raw=true)
![enter image description here](https://github.com/Mekongoabanda/ProjetIA_5A/blob/main/resultats_predict/result_predict4.png?raw=true)

## Prédiction en temps réel à travers une caméra :
![enter image description here](https://github.com/Mekongoabanda/ProjetIA_5A/blob/main/resultats_predict/real%20time_video/mask_weared_incorrect.PNG?raw=true)
![enter image description here](https://github.com/Mekongoabanda/ProjetIA_5A/blob/main/resultats_predict/real%20time_video/with_mask.PNG?raw=true)
![enter image description here](https://github.com/Mekongoabanda/ProjetIA_5A/blob/main/resultats_predict/real%20time_video/without_mask.PNG?raw=true)
![enter image description here](https://github.com/Mekongoabanda/ProjetIA_5A/blob/main/resultats_predict/real%20time_video/mask_weared_incorrect1.PNG?raw=true)
