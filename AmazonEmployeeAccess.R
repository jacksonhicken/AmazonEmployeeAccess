
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
  
amazon_test <-vroom("test.csv")

my_recipe <- recipe(ACTION ~ ., data = amazon_train) %>% 
  step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_dummy(all_nominal_predictors())

prep <- prep(my_recipe)
bake <- bake(prep, new_data = amazon_train)


