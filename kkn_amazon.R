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

amazon_train <- vroom("train.csv")
amazon_train$ACTION <- as.factor(amazon_train$ACTION)
amazon_test <-vroom("test.csv")


n <- count(amazon_test)

knn_model <- nearest_neighbor(neighbors = sqrt(n)) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")

my_recipe <- recipe(ACTION ~ ., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_normalize(all_nominal_predictors())

prep <- prep(my_recipe)
bake <- bake(prep, new_data = amazon_train)

knn_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(knn_model)

predict(knn_wf, new_data = amazon_test)





