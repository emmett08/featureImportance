library(mlr)
library(mlbench)
library(ggplot2)
library(gridExtra)
library(featureImportance)
set.seed(2023)

csv_file <- file.choose()
data <- read.csv(csv_file)
# data <- lapply(data, as.is = TRUE)

# data <- lapply( data, function(col) as.numeric( gsub("-$|\\,", "", col) ) )
# data[is.na(data)] <- 0
saveRDS(data, file = "predictive_failure.rds")

rds_object = file.choose()
predictive_failure <- readRDS(rds_object)

str(predictive_failure)

# Create regression task for mlr
node.task = makeRegrTask(data = predictive_failure, target = "predictive_node_failure")

# Specify the machine learning algorithm with the mlr package
lrn = makeLearner("regr.randomForest", ntree = 100)

# Create indices for train and test data
n = getTaskSize(node.task)
train.ind = sample(n, size = 0.6*n)
test.ind = setdiff(1:n, train.ind)

# Create test data using test indices
test = getTaskData(node.task, subset = test.ind)

# Fit model on train data using train indices
mod = train(lrn, node.task, subset = train.ind)

obs.id = sample(1:nrow(test), 20)

# Measure feature importance on test data
imp = featureImportance(mod, data = test, replace.ids = obs.id, local = TRUE)
summary(imp)
warnings()

# Plot PI and ICI curves for the lstat feature
pi.curve = plotImportance(imp, feat = "lstat", mid = "mse", individual = FALSE, hline = TRUE)
ici.curves = plotImportance(imp, feat = "lstat", mid = "mse", individual = TRUE, hline = FALSE)
grid.arrange(pi.curve, ici.curves, nrow = 1)

# Plot PI and ICI curves for the lstat feature
pi.curve = plotImportance(imp, feat = "lstat", mid = "mse", individual = FALSE, hline = TRUE)
ici.curves = plotImportance(imp, feat = "lstat", mid = "mse", individual = TRUE, hline = FALSE)
grid.arrange(pi.curve, ici.curves, nrow = 1)

rdesc = makeResampleDesc("CV", iter = 5)
res = resample(lrn, node.task, resampling = rdesc, models = TRUE)
imp = featureImportance(res, data = getTaskData(node.task), n.feat.perm = 20, local = TRUE)
summary(imp)

plotImportance(imp, feat = "lstat", mid = "mse", individual = FALSE, hline = TRUE)


