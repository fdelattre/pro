# coding=utf-8

from __future__ import print_function # import de la fonction print de python3

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.evaluation import BinaryClassificationMetrics

import os, tempfile, logging, shutil
import pandas as pd
from datetime import datetime as dt

# Gestion des arguments
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dir_path", help="Directory where to store the output", default=os.path.dirname(os.path.realpath(__file__)))
parser.add_argument("-f", "--input_file", help="Full path of the file to be processed (URI or system path)")
parser.add_argument("-s", "--separator", help="separator used in file", default=',')
parser.add_argument("-t", "--target_name", help="Name of the target", default='Target')
parser.add_argument("-c", "--cols_to_remove", help="Columns to remove if any", nargs='*', default=[])
parser.add_argument("-p", "--penalty", help="Regularization Type", default='l1')
parser.add_argument("-r", "--penalty_param", help="Penalty Parameter", type=float, default=0.8)


args = parser.parse_args()

# Définition des chemins

dir_path = args.dir_path 
input_file_name = args.input_file 
log_file_name = 'svm.log'
results_file = 'results.svm.out'

# Configuration du Log
start_time = dt.now().strftime("%b %d %Y %H:%M:%S")
logging.basicConfig(filename=os.path.join(dir_path,log_file_name),level=logging.INFO)
logging.info('Starting to Log at {}'.format(dt.now().strftime("%b %d %Y %H:%M:%S")))

# Création des contextes Spark et SQL
sc = SparkContext()
sqlContext = SQLContext(sc)
sc.setLogLevel("ERROR")
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'SparkContext created successfully')


# Définition des variables générales
separator = args.separator #','
target_name = args.target_name #'target'
cols_to_remove = args.cols_to_remove
cols_to_remove_index = []
reg_type = args.penalty
ridge_param = args.penalty_param #lambda
train_test_ratio = [0.8, 0.2]

######################################################
# Fonction de décodage des lignes en LabeledPoints
# En entrée : 
#               - la ligne à parser
#               - l'indice de la cible
#               - les indices des colonnes à ignorer
# En sortie :
#               - Un LabeledPoint
######################################################
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
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'Reading {}'.format(input_file_name))
data = sc.textFile(input_file_name)
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'done...')

header = data.first()
header_split = [item.strip('"') for item in header.split(separator)]
target_index = header_split.index(target_name)
if len(cols_to_remove) != 0:
    cols_to_remove_index = [header_split.index(col) for col in cols_to_remove] # index des variables à exclure

data = data.filter(lambda row: row != header)   # filter out header
dimensions = [data.count(), len(header.split(separator))]

# Transformation des données en LabeledPoints
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'Transformation to LabeledPoints')
points = data.map(lambda line: parsePoint(line, target_index, cols_to_remove_index))
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'done...')

# Split des données en train / test
train, test = points.randomSplit(train_test_ratio, seed=11)

# Scale des features
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'Feature scaling')
features_train = train.map(lambda x: x.features)
label_train    = train.map(lambda x: x.label)
features_test  = test.map(lambda x: x.features)
label_test     = test.map(lambda x: x.label)

scaler_model = StandardScaler(withMean=True, withStd=True).fit(features_train)
train_scale  = scaler_model.transform(features_train)
test_scale   = scaler_model.transform(features_test)

# Reconstitution des RDD de train et test
train = label_train.zip(train_scale).map(lambda x: LabeledPoint(x[0], x[1]))
test  = label_test.zip(test_scale).map(lambda x: LabeledPoint(x[0], x[1]))
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'done...')

# Entrainement du modèle
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'Model training')
model = SVMWithSGD.train(
    train, 
    iterations=100)

logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'done...')

# Sauvegarde du modèle pour usage futur
model_save_path = os.path.join(dir_path, 'model.svm')
if os.path.isdir(model_save_path):
    shutil.rmtree(model_save_path)
model.save(sc, model_save_path)
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'Model saved under {}'.format(model_save_path))



# Estimation de l'AUC sur le jeu de train
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'Compute train AUC')
predictionAndLabels_ridge_train = train.map(lambda lp: (float(model.predict(lp.features)), lp.label))
metrics_ridge_train = BinaryClassificationMetrics(predictionAndLabels_ridge_train)
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'done...')

# Estimation de l'AUC sur jeu de test
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'Compute test AUC')
predictionAndLabels_ridge_test  = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))
metrics_ridge_test  = BinaryClassificationMetrics(predictionAndLabels_ridge_test)
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'done...')

# Construction d'un DataFrame pandas contenant les coeffs
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'Create results dataframe')
variable_list = header_split[:] # Liste des variables
variable_list.remove(target_name)       # Liste des variables privée de la cible
for col in cols_to_remove:          
    variable_list.remove(col)           # Liste des variables privée des variables à exclure

coeffs = pd.DataFrame({"Variables":variable_list, "Coefficient":model.weights})
coeffs = coeffs.append(
    pd.Series({"Variables":'Intercept', "Coefficient":model.intercept}), 
    ignore_index=True)

# Version de pandas >= 0.17.0
#coeffs.sort_values(by="Coefficient", axis=1, ascending=False, inplace=True)

# Version de pandas < 0.17.0
coeffs = coeffs.sort("Coefficient", ascending=False)
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'done...')

end_time = dt.now().strftime("%b %d %Y %H:%M:%S")

# Ecriture d'un fichier de sortie
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'Writing results')
text_file = open(os.path.join(dir_path, results_file), 'w')

print("-------------------------------------------------------------------------------------------------", file=text_file)
print("-------------------------------------------------------------------------------------------------", file=text_file)

print("Fichier traité: {}".format(input_file_name), file=text_file)

print("Dimensions : {}".format(dimensions), file=text_file)
print("Cible : {}".format(target_name), file=text_file)
print("Colonnes exclues : {}".format(cols_to_remove), file=text_file)
print("\n", file=text_file)
print("Début du traitement : {}".format(start_time), file=text_file)
print("Fin du traitement : {}".format(end_time), file=text_file)
print("-------------------------------------------------------------------------------------------------", file=text_file)
print("-------------------------------------------------------------------------------------------------", file=text_file)
print("Régression logistique pénalisée {} avec lambda = {}".format(reg_type, ridge_param), file=text_file)
print("Modéle sauvegardé sous {}".format(model_save_path), file=text_file)
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
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'done...')


# Close Spark Context
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'closing Spark Context')
sc.stop()
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'done...')

