
library(tidymodels)
library(tidyverse)
library(tidymodels)
library(vroom)
library(patchwork)
library(poissonreg)
library(parsnip)
library(glmnet)
library(rpart)
library(ranger)
library(stacks)
library(dbarts)
library(embed)
library(kknn)


# load data ----------------------------------------------------------------

amazon_train <- vroom("train.csv")
amazon_train$ACTION <- as.factor(amazon_train$ACTION)

amazon_test <-vroom("test.csv")

my_recipe <- recipe(ACTION ~ ., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_dummy(all_nominal_predictors())

my_recipe <- recipe(ACTION ~ ., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = .9)

my_recipe <- recipe(ACTION~., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(),fn= factor) %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  #step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = .9)


# logistic regression -----------------------------------------------------

logistic_model <- logistic_reg() %>% 
  set_engine("glm")

amazon_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model()

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

folds <- vfold_cv(amazon_train, v =5, repeats = 1)

CV_results <- amazon_workflow %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid, 
            metrics = metric_set(roc_auc))

bestTune <- CV_results %>% 
  select_best("roc_auc")

final_wf <- amazon_workflow %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = amazon_train)


amazon_predictions <- predict(amazon_workflow,
                              new_data = amazon_test,
                              type = "prob")

kaggle_submission <- amazon_predictions %>%
  bind_cols(., amazon_test) %>%
  rename(Action = .pred_1) %>% 
  select(id, Action)

vroom_write(x=kaggle_submission, file="./amazon_logistic1_pca.csv", delim=",")


# penalized regression ----------------------------------------------------

my_recipe1 <- recipe(ACTION ~ ., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  # step_lencode_mixed(all_numeric_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors())

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 3)

folds <- vfold_cv(amazon_train, v = 3, repeats=1)

CV_results <- amazon_workflow %>%
  tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best()

final_wf <- amazon_workflow %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = amazon_train)

amazon_predictions <- predict(final_wf, new_data = amazon_test,
                       type = "prob")

kaggle_submission <- amazon_predictions %>%
  bind_cols(., amazon_test) %>%
  rename(Action = .pred_1) %>% 
  select(id, Action)

vroom_write(x=kaggle_submission, file="./amazon_logistic1_pca.csv", delim=",")


# knn ---------------------------------------------------------------------

n <- count(amazon_test)

knn_model <- nearest_neighbor(neighbors = sqrt(n)) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")

my_recipe <- recipe(ACTION ~ ., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

my_recipe <- recipe(ACTION ~ ., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_predictors()) %>% 
  step_pca(all_predictors(), threshold = .9)

knn_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(knn_model) %>% 
  fit(data = amazon_train)

kknn_preds <- predict(knn_wf, new_data = amazon_test, type = "prob")

kaggle_submission <- kknn_preds %>%
  bind_cols(., amazon_test) %>%
  rename(Action = .pred_1) %>% 
  select(id, Action)

vroom_write(x=kaggle_submission, file="./amazon_kknn_pca.csv", delim=",")


# random forest -----------------------------------------------------------

rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

rf_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(rf_model)


tuning_grid <- grid_regular(mtry(range = c(1,9)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(amazon_train, v = 10, repeats = 1)

CV_results <- rf_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best()

final_wf <- rf_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = amazon_train)

rf_preds <- predict(final_wf,
                    new_data = amazon_test,
                    type = "prob")

kaggle_submission <- rf_preds %>%
  bind_cols(., amazon_test) %>%
  rename(Action = .pred_1) %>% 
  select(id, Action)

vroom_write(x=kaggle_submission, file="./amazon_rf_final2.csv", delim=",")


# naive bayes -------------------------------------------------------------

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

nb_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5)

folds <- vfold_cv(amazon_train, v = 5, repeats=1)


CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

bestTune <- CV_results %>%
  select_best()

final_wf <- nb_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data = amazon_train)


amazon_predictions <- predict(final_wf, new_data = amazon_test,
                              type = "prob")

kaggle_submission <- amazon_predictions %>%
  bind_cols(., amazon_test) %>%
  rename(Action = .pred_1) %>% 
  select(id, Action)

vroom_write(x=kaggle_submission, file="./amazon_nb_pca_smote2.csv", delim=",")

