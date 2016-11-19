library(sparklyr)
library(dplyr)
library(readr)
 
german <- readr::read_csv("~/code/GermanCredit/german_credit.csv")
scon <- spark_connect(master = "local")
sparklyr::spark_version(scon)

german_sp <- dplyr::copy_to(df = german, dest = scon)

german_sp %>%
  group_by(Creditability)%>%
  summarise(mean(Age_years), min(Age_years), max(Age_years))

partitions <- german_sp %>%
  sdf_partition(training = 0.8, test = 0.2, seed = 1099)

feature_names <- setdiff(dimnames(german_sp)[[2]], "Creditability")
fit <- partitions$training %>%
  ml_logistic_regression(response = "Creditability", features = feature_names, alpha = 0, lambda=0.01)

sparklyr::ml_save(fit, "~/code/GermanCredit/model.sparlyr")

pred <- sdf_predict(fit, partitions$test)
ml_binary_classification_eval(pred, "Creditability", "prediction", metric = "areaUnderROC")
