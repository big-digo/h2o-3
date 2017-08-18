setwd(normalizePath(dirname(R.utils::commandArgs(asValues=TRUE)$"f")))
source("../../../scripts/h2o-r-test-setup.R")

# TO DO: Replicate for gaussian

stackedensemble.base_model_training_frame.test <- function() {
  
  # This test checks the following (for binomial classification):
  # 
  # 1) That passing in base models that use different subsets of 
  #    the features (but same training_frame) works.
  # 2) TO DO: That passing in base models that use training frames with different nrows fails.
  # 3) TO DO: That passing in base models that use different training frames fails.
  
  train <- h2o.uploadFile(locate("smalldata/testng/higgs_train_5k.csv"), 
                          destination_frame = "higgs_train_5k")
  
  y <- "response"
  x <- setdiff(names(train), y)
  train[,y] <- as.factor(train[,y])
  nfolds <- 3
  
  # Train & Cross-validate a GBM
  my_gbm <- h2o.gbm(x = x[1:10], 
                    y = y, 
                    training_frame = train, 
                    distribution = "bernoulli",
                    ntrees = 10, 
                    nfolds = nfolds, 
                    fold_assignment = "Modulo",
                    keep_cross_validation_predictions = TRUE,
                    seed = 1)
  
  
  # Train & Cross-validate a RF
  my_rf <- h2o.randomForest(x = x[14:20],
                            y = y, 
                            training_frame = train,
                            distribution = "bernoulli",
                            ntrees = 10, 
                            nfolds = nfolds, 
                            fold_assignment = "Modulo",
                            keep_cross_validation_predictions = TRUE,
                            seed = 1)
  
  
  # Train a stacked ensemble using the GBM and RF above
  stack1 <- h2o.stackedEnsemble(x = x, 
                                y = y, 
                                training_frame = train,
                                base_models = list(my_gbm@model_id, my_rf@model_id))
  
  
  # Train a stacked ensemble using the GBM and RF above (no x)
  stack2 <- h2o.stackedEnsemble(y = y, 
                                training_frame = train,
                                base_models = list(my_gbm@model_id, my_rf@model_id))
  
  # Eval train AUC to assess equivalence
  expect_equal(h2o.auc(stack1), h2o.auc(stack2))
  
}

doTest("Stacked Ensemble base_models training_frame Test", stackedensemble.base_model_training_frame.test)
