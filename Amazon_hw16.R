
# libraries

library(tidyverse)
library(tidymodels)
library(vroom)
library(poissonreg)
library(glmnet)
library(embed)


# data

amazon_train <- vroom("train.csv")
amazon_train$ACTION <- as.factor(amazon_train$ACTION)
amazon_test <-vroom("test.csv")


# logistic model ----------------------------------------------------------

# recipe
my_recipe <- recipe(ACTION ~ ., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .01) %>% 
  step_dummy(all_nominal_predictors())

# model

logistic_model <- logistic_reg() %>% 
  set_engine("glm")

# workflow

amazon_workflow <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(logistic_model) %>% 
  fit(data = amazon_train)

amazon_predictions <- predict(amazon_workflow, new_data = amazon_test,
                              type = "prob")

kaggle_submission <- amazon_predictions %>%
  bind_cols(., amazon_test) %>%
  rename(Action = .pred_1) %>% 
  select(id, Action)

vroom_write(x=kaggle_submission, file="amazon_logistic_preds.csv", delim=",")




# penalized model ---------------------------------------------------------


# recipe

my_recipe1 <- recipe(ACTION ~ ., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_numeric_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors())

# model

penalized_model <- logistic_reg(mixture = .8, penalty = .01) %>% 
  set_engine("glm")

# workflow

amazon_workflow_penalized <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(penalized_model) %>% 
  fit(data = amazon_train)

amazon_predictions_pen <- predict(amazon_workflow_penalized, new_data = amazon_test,
                              type = "prob")

kaggle_submission2 <- amazon_predictions_pen %>%
  bind_cols(., amazon_test) %>%
  rename(Action = .pred_1) %>% 
  select(id, Action)

vroom_write(x=kaggle_submission2, file="amazon_logistic_preds_pen.csv", delim=",")


