# coding=utf-8

from __future__ import print_function # import de la fonction print de python3

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionModel
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

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
parser.add_argument("-r", "--ridge_param", help="Ridge Param", type=float, default=0.01)
parser.add_argument("-l", "--lasso_param", help="Lasso Param", type=float, default=0.8)


args = parser.parse_args()

# Définition des chemins
dir_path = args.dir_path 
input_file_name = args.input_file 
log_file_name = 'reglog.log'
results_file = 'results.reglog.out'

# Configuration du Log
start_time = dt.now().strftime("%b %d %Y %H:%M:%S")
logging.basicConfig(filename=os.path.join(dir_path,log_file_name),level=logging.INFO)
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'Starting to Log')

# Création des contextes Spark et SQL
sc = SparkContext()
sqlContext = SQLContext(sc)
sc.setLogLevel("ERROR")
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'SparkContext created successfully')


# Définition des variables générales
separator = args.separator
target_name = args.target_name
cols_to_remove = args.cols_to_remove
cols_to_remove_index = []
ridge_param = args.ridge_param #lasso
lasso_param = args.lasso_param #ridge
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
#######################################################


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

#points_df = points.toDF()
#target_ratio = points_df.groupBy("label").count()

# Split des données en train / test
train, test = points.randomSplit(train_test_ratio, seed=11)

# Définition du modèle
model =  LogisticRegression(
    maxIter = 10, 
    regParam = ridge_param, 
    elasticNetParam=lasso_param)

# Définition d'un Pipeline
pipeline = Pipeline(stages=[model])
lrFit = pipeline.fit(train.toDF())

logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'LogisticRegression model trained')

# Sauvegarde du modèle pour usage futur
model_save_path = os.path.join(dir_path, 'model.reglog')
if os.path.isdir(model_save_path):
    shutil.rmtree(model_save_path)
#lrFit.stages[0].save(sc, model_save_path)
#logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'Model saved under {}'.format(model_save_path))



# Estimation de l'AUC sur le jeu de train
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'Compute train AUC')
predictionAndLabels_ridge_train = lrFit.transform(train.toDF())
evaluator = BinaryClassificationEvaluator()
auc_train = evaluator.evaluate(predictionAndLabels_ridge_train)
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'done...')


# Estimation de l'AUC sur jeu de test
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'Compute test AUC')
predictionAndLabels_ridge_test  = lrFit.transform(test.toDF())
evaluator_test = BinaryClassificationEvaluator()
auc_test = evaluator_test.evaluate(predictionAndLabels_ridge_test)
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'done...')

# Construction d'un DataFrame pandas contenant les coeffs
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'Create results dataframe')
variable_list = header_split[:] # Liste des variables
variable_list.remove(target_name)       # Liste des variables privée de la cible
for col in cols_to_remove:          
    variable_list.remove(col)           # Liste des variables privée des variables à exclure

coeffs = pd.DataFrame(
    {"Variable":variable_list, "Coefficient":lrFit.stages[0].coefficients})

#Ajout de l'intercept
coeffs = coeffs.append(
    pd.Series({"Variable":'Intercept', "Coefficient":lrFit.stages[0].intercept}), 
    ignore_index=True)

# Version de pandas >= 0.17.0
#coeffs.sort_values(by="Coefficient", axis=1, ascending=False, inplace=True)

# Version de pandas < 0.17.0
coeffs = coeffs.sort("Coefficient", ascending=False)
coeffs_non_nuls = coeffs[coeffs['Coefficient'] != 0]
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'done...')

end_time = dt.now().strftime("%b %d %Y %H:%M:%S")


# Ecriture d'un fichier de sortie
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'Writing results')
text_file = open(os.path.join(dir_path, results_file), 'w')

print(dt.now().strftime("%b %d %Y %H:%M:%S"), file=text_file)
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
print("Régression logistique pénalisée elastic net avec lambda = {} et lasso = {}".format(ridge_param, lasso_param), file=text_file)
print("Modéle sauvegardé sous {}".format(model_save_path), file=text_file)
print("-------------------------------------------------------------------------------------------------", file=text_file)
print("-------------------------------------------------------------------------------------------------", file=text_file)
print("AUC jeu de train = {}".format(auc_train), file = text_file)
print("AUC jeu de test = {}".format(auc_test), file = text_file)
print("Ratio Train / Test : {}".format(train_test_ratio), file=text_file)
print("-------------------------------------------------------------------------------------------------", file=text_file)
print("---------------------------------COEFFICIENTS----------------------------------------------------", file=text_file)
print("-------------------------------------------------------------------------------------------------", file=text_file)
print("Nombre de coefficients non nuls : {}".format(coeffs_non_nuls.shape[0]), file=text_file)
print(coeffs_non_nuls.to_string(), file=text_file)
# Close file
text_file.close()
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'done...')

# Close Spark Context
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'closing Spark Context')
sc.stop()
logging.info('{} : '.format(dt.now().strftime("%b %d %Y %H:%M:%S")) + 'done...')
