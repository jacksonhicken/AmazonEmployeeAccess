
##### AMAZON EMPLOYEE ACCESS - STAT 348 #####


# packages ----------------------------------------------------------------

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


# load data ----------------------------------------------------------------

amazon_train <- vroom("train.csv")
amazon_train$ACTION <- as.factor(amazon_train$ACTION)

amazon_test <-vroom("test.csv")

my_recipe <- recipe(ACTION ~ ., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_dummy(all_nominal_predictors())

prep <- prep(my_recipe)
bake <- bake(prep, new_data = amazon_train)


# logistic regression -----------------------------------------------------

logistic_model <- logistic_reg(mixture = .8, penalty = .01) %>% 
  set_engine("glm")

amazon_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(logistic_model)

view(amazon_predictions) <- predict(amazon_workflow,
                              new_data = amazon_test,
                              type = "prob")

kaggle_submission <- amazon_predictions %>%
  bind_cols(., amazon_test) %>%
  rename(Action = .pred_1) %>% 
  select(id, Action)

vroom_write(x=kaggle_submission, file="./amazon_logistic2.csv", delim=",")

# penalized ######################

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


