# Lancement du la reglog avec Spark

./bin/spark-submit --master local[*] \
  ../GermanCredit/logreg.py \
  -f ~/code/GermanCredit/german_credit.csv \
  -t Creditability \
  -s ;

