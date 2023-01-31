# nstall.packages("caret")
library(caret)
# install.packages("ggplot2")
library(ggplot2)
# install.packages("lattice")
library(lattice)
# install.packages("tidyverse")
library(tidyverse)
# install.packages("nnet")
library(nnet)

# install.packages("MASS")
library(MASS)

# install.packages("glmnet")
library(glmnet)

csv_file <- file.choose()
csv_data <- read.csv(csv_file)

saveRDS(csv_data, file = "predictive_node_failure_filtered.Rds")

rds_file <- file.choose()
data <- readRDS(rds_file)
# data[data == 0] <- NA

set.seed(430)
train_idx = createDataPartition(data$predictive_node_failure, p = 0.75, list = FALSE)
sim_trn = data[train_idx, ]
sim_tst = data[-train_idx, ]
summary(sim_trn)
summary(sim_tst)

log_model <- glm(predictive_node_failure ~ ., data = sim_trn, family = "binomial")
predictions <- predict(log_model, newdata = sim_tst, type = "response")

accuracy <- mean(predictions == sim_tst$predictive_node_failure)
print(accuracy)

ggplot(data = sim_tst, aes(x = predictions, y = predictive_node_failure)) + geom_point(alpha = 0.2) + geom_abline(intercept = 0, slope = 1) + theme_bw() + labs(title = "Logistic model predictions vs. true values")

