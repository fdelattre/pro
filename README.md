# Lancement de la reglog avec Spark


<code>export SPARK_HOME=/home/francois/code/spark-1.6.1-bin-hadoop2.6</code>

<code>$SPARK_HOME/bin/spark-submit --master local[*]
  logreg.py
  -f german_credit.csv
  -t Creditability
  -s ","
  -c Purpose
  -r 0.05
  </code>

Liste des options :
* -d chemin/vers/repertoire/de/sortie (par défaut le répertoire où est stocké le fichier logreg.py)
* -f nom_du_fichier.csv
* -s separateur (par défaut ',')
* -t cible
* -c col1 col2 col 3 : la liste des colonnes à exclure
* -r 0.01 la paramètre de pénalité (par défaut 0.01)
