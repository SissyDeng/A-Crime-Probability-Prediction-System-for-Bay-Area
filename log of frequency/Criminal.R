############################library######################################
library(softImpute)
library(randomForest)
library(ranger)
library(dplyr)
library(tidyverse)
library(reshape2)
library(stringr)
library(ggplot2)
library(BBmisc)
library(tm)
library(MASS)
library(caTools)
library(rpart)
library(rpart.plot)
library(caret)
library(gbm)

############################function######################################
OSR2 <- function(predictions, train, test) {
  SSE <- sum((test - predictions)^2)
  SST <- sum((test - mean(train))^2)
  r2 <- 1 - SSE/SST
  return(r2)
}

# prediction
pred <- function(alpha,beta,test){
  pred_biScale <- rep(0,nrow(test))
  for(i in 1:nrow(test)){
    pred_biScale[i] <- alpha[test[i,1]] + beta[test[i,2]]
  }
  return(pred_biScale)
}

# append data
train.new <- inner_join(train,crime,by="crimeID")
val1.new <- inner_join(val1,crime,by="crimeID")
val2.new <- inner_join(val2,crime,by="crimeID")
test.new <- inner_join(test,crime,by="crimeID")
# place in order
train.new$time <- as.factor(train.new$time)
val1.new$time <- as.factor(val1.new$time)
val2.new$time <- as.factor(val2.new$time)
test.new$time <- as.factor(test.new$time)

############################import data######################################
criminal <- read.csv("CRM.csv",header = TRUE)
crime <- read.csv("crime.csv",header = TRUE)
crime <- crime %>%  
  mutate(crimeID = as.numeric(crimeID))

#criminal$ratings = normalize(criminal$Freq, method = "range", range = c(0, 10), margin = 1L, on.constant = "quiet")

ggplot(data=criminal, mapping=aes(x=Freq))+
  geom_bar(stat="count",width=2)

# Log
criminal$Freq = log(criminal$Freq)
# distribution plot
ggplot(data=criminal, mapping=aes(x=Freq))+
  geom_bar(stat="count",width=0.1)

set.seed(123)
train.ids <- sample(nrow(criminal), 0.92*nrow(criminal))
train <- criminal[train.ids,]
test <- criminal[-train.ids,]

# split training into real training and validation set
val1.ids <- sample(nrow(train), (4/92)*nrow(train))
val1 <- train[val1.ids,]
train <- train[-val1.ids,]

val2.ids <- sample(nrow(train), (4/88)*nrow(train))
val2 <- train[val2.ids,]
train <- train[-val2.ids,]

############################CF######################################
# Construct an incomplete training set Freq matrix
mat.train <- Incomplete(train$neighborID, train$crimeID, train$Freq)
summary(train)

# CF model
set.seed(345)
mae.vals = rep(NA, 60)
for (rnk in seq_len(60)) {
  print(str_c("Trying rank.max = ", rnk))
  mod <- softImpute(mat.train, rank.max = rnk, lambda = 0, maxit = 1000)
  preds <- impute(mod, val1$neighborID, val1$crimeID) %>% pmin(5) %>% pmax(1)
  mae.vals[rnk] <- mean(abs(preds - val1$Freq))
}

mae.val.df <- data.frame(rnk = seq_len(60), mae = mae.vals)
ggplot(mae.val.df, aes(x = rnk, y = mae)) + geom_point(size = 3) + 
  ylab("Validation MAE") + xlab("Number of Archetypal Neighbors") + 
  theme_bw() + theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

# choose k = 1
set.seed(345)
mod.final <- softImpute(mat.train, rank.max = 1, lambda = 0, maxit = 1000)
preds <- impute(mod.final, test$neighborID, test$crimeID) %>% pmin(5) %>% pmax(1)

mean(abs(preds - test$Freq))/range(test$Freq)[2]
sqrt(mean((preds - test$Freq)^2))/range(test$Freq)[2]
OSR2(preds, train$Freq, test$Freq)

############################Linear regression######################################
# linear regression model
set.seed(123)
mod1 <- lm(Freq ~ . -crimeID -Incident.Category -Analysis.Neighborhood -Incident.Time -neighborID, data = train.new)
summary(mod1)
pred1 <- predict(mod1, newdata = test.new) 

mean(abs(pred1 - test.new$Freq))/range(test.new$Freq)[2]
sqrt(mean((pred1 - test.new$Freq)^2))/range(test.new$Freq)[2]
OSR2(pred1, train.new$Freq, test.new$Freq)

############################Random forest######################################
# random forests
set.seed(123)
rf.mod <- ranger(Freq ~ . -crimeID -Incident.Category -Analysis.Neighborhood -Incident.Time -neighborID, 
                 data = train.new, 
                 mtry = 1, 
                 num.trees = 500,
                 verbose = TRUE)

preds.rf <- predict(rf.mod, data = test.new)
preds.rf <- preds.rf$predictions
mean(abs(preds.rf - test.new$Freq))/range(test.new$Freq)[2]
sqrt(mean((preds.rf - test.new$Freq)^2))/range(test.new$Freq)[2]
OSR2(preds.rf, train.new$Freq, test.new$Freq)

############################Boosting######################################
# Boosting
set.seed(123)
mod.boost <- gbm(Freq ~ time+Total.Population+Male.Female+mile.sq.mile.+Median.Age+Median.Household.Income...,
                 data = train.new,
                 distribution = "gaussian",
                 n.trees = 1000,
                 shrinkage = 0.001,
                 interaction.depth = 2)

pred.boost <- predict(mod.boost, newdata = test.new, n.trees=1000)
summary(mod.boost)
mean(abs(pred.boost - test.new$Freq))/range(test.new$Freq)[2]
sqrt(mean((pred.boost - test.new$Freq)^2))/range(test.new$Freq)[2]
OSR2(pred.boost, train.new$Freq, test.new$Freq)

############################Blending######################################
# Blending
val.preds.cf <- impute(mod.final, val2.new$neighborID, val2.new$crimeID) %>% pmin(5) %>% pmax(1)
val.preds.lm <- predict(mod1, newdata = val2.new)
val.preds.rf <- predict(rf.mod, newdata = val2.new)$predictions
val.preds.boost <- predict(mod.boost, newdata = val2.new, n.trees=1000)

# Build validation set data frame
val.blending_df = data.frame(Freq = val2.new$Freq, 
                             cf_preds = val.preds.cf, 
                             lm_preds = val.preds.lm,
                             rf_preds = val.preds.rf,
                             rf_boost = val.preds.boost)

# Train blended model
blend.mod = lm(Freq ~ . -1, data = val.blending_df)
summary(blend.mod)

# Get predictions on test set
test.preds.cf <- impute(mod.final, test.new$neighborID, test.new$crimeID) %>% pmin(5) %>% pmax(1)
test.preds.lm <- predict(mod1, newdata = test.new)
test.preds.rf <- predict(rf.mod, data = test.new)$predictions
test.preds.boost <- predict(mod.boost, newdata = test.new, n.trees=1000)

test.blending_df = data.frame(Freq = test.new$Freq, 
                              cf_preds = test.preds.cf, 
                              lm_preds = test.preds.lm,
                              rf_preds = test.preds.rf,
                              rf_boost = test.preds.boost)

test.preds.blend <- predict(blend.mod, newdata = test.blending_df)

mean(abs(test.preds.blend - test.new$Freq))/range(test.new$Freq)[2]
sqrt(mean((test.preds.blend - test.new$Freq)^2))/range(test.new$Freq)[2]
OSR2(test.preds.blend, train.new$Freq, test.new$Freq)

############################CART######################################
# Cross-validated CART model
set.seed(123)
train.cart = train(Freq ~ time+Total.Population+Male.Female+mile.sq.mile.+Median.Age+Median.Household.Income...,
                   data = train.new,
                   method = "rpart",
                   tuneGrid = data.frame(cp=seq(0, 1e-02, 1e-04)),
                   trControl = trainControl(method="cv", number=5),
                   metric = "RMSE")
train.cart
train.cart$results

ggplot(train.cart$results, aes(x = cp, y = RMSE)) + 
  geom_point(size = 2) + 
  geom_line() + 
  ylab("CV RMSE") + 
  theme_bw() + 
  theme(axis.title=element_text(size=18), axis.text=element_text(size=18))

mod.cart = train.cart$finalModel
prp(mod.cart)
test.cart = as.data.frame(model.matrix(Freq ~ time+Total.Population+Male.Female+mile.sq.mile.+Median.Age+Median.Household.Income..., data=test.new)) 
pred.cart = predict(mod.cart, newdata = test.cart)
mean(abs(pred.cart - test.new$Freq))/range(test.new$Freq)[2]
sqrt(mean((pred.cart - test.new$Freq)^2))/range(test.new$Freq)[2]
OSR2(pred.cart, train.new$Freq, test.new$Freq)