# Mini Kaggle API

## Description
Créez une API à laquelle on peut envoyer des prédictions faites sur un dataset de test donné, via un "submission file" ayant les mêmes spécifications que sur Kaggle, et qui renvoie le score sur le dataset en question.

* On se base sur le challenge Give Me Some Credit de Kaggle. 
* Création de l'API avec Flask.

## Etapes
Pour obtenir un dataset de test sur lequel les output sont connues, le dataset de train est partagé en 2  
Sauvegarde du dataset de test dans un fichier test2.csv  
Ensuite on fournit un fichier contenant des prédictions sur ce dataset **test2-predictions.csv**.

## Test
Les commandes suivantes sont à exécuter afin de tester le projet:

```
pip install -r requirements.txt

export FLASK_APP=api.py
flask run

curl --request POST \
  --url 'http://localhost:5000/submit' \
  --header 'accept: multipart/form-data' \
  -F 'file=@test2-predictions.csv'
```