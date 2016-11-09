from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics

import pandas as pd
import os
import logging
from datetime import datetime as dt

# Configuration du Log
logging.basicConfig(filename='D:\\Users\\s36733\\Documents\\Projets\\TestSpark\\logreg.log',level=logging.DEBUG)
logging.info('Starting to Log at {}'.format(dt.now().strftime("%b %d %Y %H:%M:%S")))

sc = SparkContext()
logging.info('SparkContext created successfully')

sc.setLogLevel("ERROR")

# Définition des variables générales
filepath = "D:\\Users\\s36733\\Documents\\Projets\\TestSpark\\german_credit.csv"
separator = ','
target_name = 'Creditability'
cols_to_remove = ['Purpose', 'Telephone']
cols_to_remove_index = []

#filepath = "D:\\Users\\s36733\\Documents\\Projets\\ScoreB2BNord\\dtm_sample2.csv"
#separator = ';'
#target_name = 'cible_reut2'
#cols_to_remove = ['idcli_horus', 'nbreut2', 'reut_ff', 'reut_sof', 'reut_cacf']
#cols_to_remove_index = []

train_test_ratio = [0.8, 0.2]
ridge_param = 0.001 #lambda

# Fonction de décodage des lignes en LabeledPoints
def parsePoint(line, target_index, cols_to_remove_index):
    values = [float(s) for s in line.split(separator)]
    target = values[target_index]
    if  target == -1:   # Convert -1 labels to 0 for MLlib
        target = 0
    
    local_col_to_remove = cols_to_remove_index[:] # shallow copy pour ne pas modifier la variable cols_to_remove_index
    local_col_to_remove.append(target_index)
    
    for i in sorted(local_col_to_remove, reverse=True):
        values.pop(i)
    return LabeledPoint(target, values)

# Lecture du fichier
data = sc.textFile(filepath)
logging.info('File {} read'.format(filepath))

header = data.first()
header_split = [item.strip('"') for item in header.split(separator)]
target_index = header_split.index(target_name)
if len(cols_to_remove) != 0:
    cols_to_remove_index = [header_split.index(col) for col in cols_to_remove] # index des variables à exclure

data = data.filter(lambda row: row != header)   # filter out header
dimensions = [data.count(), len(header.split(separator))]

# Transformation des données en LabeledPoints
points = data.map(lambda line: parsePoint(line, target_index, cols_to_remove_index))
logging.info('Labelled Points created')

# Split des données en train / test
train, test = points.randomSplit(train_test_ratio, seed=11)

# Entrainement du modèle
model = LogisticRegressionWithLBFGS.train(
    train, 
    intercept = True, 
    regType = 'l2', 
    regParam = ridge_param)
logging.info('LogisticRegressionWithLBFGS model trained')

model.save("LogisticRegression")


# Estimation de l'AUC sur le jeu de train
predictionAndLabels_ridge_train = train.map(lambda lp: (float(model.predict(lp.features)), lp.label))
metrics_ridge_train = BinaryClassificationMetrics(predictionAndLabels_ridge_train)

# Estimation de l'AUC sur jeu de test
predictionAndLabels_ridge_test  = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))
metrics_ridge_test  = BinaryClassificationMetrics(predictionAndLabels_ridge_test)

# Construction d'un DataFrame pandas contenant les coeffs
variable_list = header.split(separator) # Liste des variables
variable_list.remove(target_name)       # Liste des variables privée de la cible
for col in cols_to_remove:          
    variable_list.remove(col)           # Liste des variables privée des variables à exclure

coeffs = pd.DataFrame({"Variables":variable_list, "Coefficient":model.weights})
coeffs = coeffs.append(
    pd.Series({"Variables":'Intercept', "Coefficient":model.intercept}), 
    ignore_index=True)


# Ecriture d'un fichier de sortie

text_file = open('D:\\Users\\s36733\\Documents\\Projets\\TestSpark\\results.out', 'w')

text_file.write(dt.now().strftime("%b %d %Y %H:%M:%S"))
print(dt.now().strftime("%b %d %Y %H:%M:%S"), file=text_file)
print("-------------------------------------------------------------------------------------------------", file=text_file)
print("-------------------------------------------------------------------------------------------------", file=text_file)

print("Fichier traité: {}".format(filepath), file=text_file)

print("Dimensions : {}".format(dimensions), file=text_file)
print("Cible : {}".format(target_name), file=text_file)
print("Colonnes exclues : {}".format(cols_to_remove), file=text_file)
print("\n", file=text_file)
print("-------------------------------------------------------------------------------------------------", file=text_file)
print("-------------------------------------------------------------------------------------------------", file=text_file)
print("Régression logistique pénalisée L2 avec lambda = {}".format(ridge_param), file=text_file)
# print("Modéle sauvegardé sous {}".format(save_file), file=text_file)
print("-------------------------------------------------------------------------------------------------", file=text_file)
print("-------------------------------------------------------------------------------------------------", file=text_file)
print("AUC jeu de train = {}".format(metrics_ridge_train.areaUnderROC), file = text_file)
print("AUC jeu de test = {}".format(metrics_ridge_test.areaUnderROC), file = text_file)
print("Ratio Train / Test : {}".format(train_test_ratio), file=text_file)
print("-------------------------------------------------------------------------------------------------", file=text_file)
print("---------------------------------COEFFICIENTS----------------------------------------------------", file=text_file)
print("-------------------------------------------------------------------------------------------------", file=text_file)

print(coeffs.to_string(), file=text_file)

text_file.close()


# Close Spark Context
sc.stop()
