# Lancement de la reglog avec Spark

export SPARK_HOME=/home/francois/code/spark-1.6.1-bin-hadoop2.6

$SPARK_HOME/bin/spark-submit --master local[*] \
  logreg.py \
  --input_file german_credit.csv \
  --target_name Creditability \
  --separator ;

