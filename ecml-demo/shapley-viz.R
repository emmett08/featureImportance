library(checkpoint)
checkpoint("2018-03-01", project = "ecml-demo/helper", forceProject = TRUE)
source("ecml-demo/helper/packages.R")
install()
library(featureImportance)
library(ggplot2)
library(gridExtra)


path = "ecml-demo/application_shapley_simulation"
unlink(path, recursive = TRUE)
reg = makeExperimentRegistry(
file.dir = path,
packages = c("featureImportance"),
source = paste0("ecml-demo/", c("helper/functions.R", "helper/packages.R")),
seed = 123)

reg$cluster.functions = makeClusterFunctionsSocket(30)

res = readRDS("application_shapley_simulation.Rds")
# use shorter learner name
res[, learner := factor(gsub("regr.", "", learner))]
# compute ratio of the importance values w.r.t feature "V3"
res[, ratio := mse/mse[feature == "V3"], by = c("method", "learner", "repl")]

### Plot simulation results of all 500 repititions
shap = subset(res, method %in% c("pfi.diff", "pfi.ratio", "shapley") & feature != "V3")
new.names = setNames(expression(X[1]/X[3], X[2]/X[3]), c("V1", "V2"))

pp = ggplot(data = shap, aes(x = feature, y = ratio)) + 
  geom_boxplot(aes(fill = method), lwd = 0.2, outlier.size = 0.8) + 
  facet_grid(. ~ learner, scales = "free") +
  scale_fill_grey(labels = c("PFI (Diff.)", "PFI (Ratio)", "SFIMP"), start = 0.4, end = 0.9) +
  scale_x_discrete(labels = new.names) +
  labs(title = "(b) Simulation with 500 repetitions", 
    x = "Features involved to compute the ratio", y = "Value of the ratio")

### Plot example of an individual repetition (2nd replication)
shap2 = subset(res, repl == 2 & method %in% c("shapley", "geP"))
# reorder features for plotting
feat.order = c("V3", "V2", "V1", "geP")
shap2$feature = factor(shap2$feature, levels = feat.order)
# change sign
shap2[, mse := round(ifelse(method == "geP", mse, -mse), 2)]
# add column containing proportion of explained importance
shap2[, perc := ifelse(feature == "geP", NA, mse/sum(mse[feature != "geP"])), by = "learner"]
# add column containing drop in MSE + proportion of explained importance
shap2[, lab := ifelse(feature == "geP", mse, paste0(mse, " (", round(perc*100, 0), "%)"))]

col = c(gray.colors(3, start = 0.4, end = 0.9), hcl(h = 195, l = 65, c = 100))
col = setNames(col, feat.order)
legend = c("V1" = bquote(phi[1]), "V2" = bquote(phi[2]), 
  "V3" = bquote(phi[3]), "geP" = bquote(widehat(GE)[P]))

pp2 = ggplot(shap2, aes(x = learner, y = mse, fill = feature)) + 
  geom_bar(stat = "identity", colour = "white", pos = "stack") + 
  geom_text(aes(label = lab), position = position_stack(vjust = 0.5), size = 3) + 
  coord_flip() + 
  scale_fill_manual(values = col, name = " performance \n explained by", labels = legend) +
  labs(title = "(a) Comparing the model performance and SFIMP values across different models", 
    x = "", y = "performance (MSE)")

grid.arrange(pp2, pp, heights = c(3, 5))
