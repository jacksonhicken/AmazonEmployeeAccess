
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
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_dummy(all_nominal_predictors())

prep <- prep(my_recipe)
bake <- bake(prep, new_data = amazon_train)


# logistic regression -----------------------------------------------------

logistic_model <- logistic_reg() %>% 
  set_engine("glm")

amazon_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(logistic_model) %>% 
  fit(data = amazon_train)

view(amazon_predictions) <- predict(amazon_workflow,
                              new_data = amazon_test,
                              type = "prob")

kaggle_submission <- amazon_predictions %>%
  bind_cols(., amazon_test) %>%
  rename(Action = .pred_1) %>% 
  select(id, Action)

vroom_write(x=kaggle_submission, file="./amazon_logistic1.csv", delim=",")


