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

amazon_train <- vroom("train.csv")
amazon_train$ACTION <- as.factor(amazon_train$ACTION)

amazon_test <-vroom("test.csv")

my_recipe <- recipe(ACTION ~ ., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_dummy(all_nominal_predictors())

rf_model <- rand_forest(mtry = 100,
                        min_n = 10,
                        trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("classification")

amazon_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(rf_model) %>% 
  fit(data = amazon_train)

rf_preds <- predict(amazon_workflow,
                    new_data = amazon_test,
                    type = "prob")

kaggle_submission <- rf_preds %>%
  bind_cols(., amazon_test) %>%
  rename(Action = .pred_1) %>% 
  select(id, Action)

vroom_write(x=kaggle_submission, file="./amazon_rf.csv", delim=",")

