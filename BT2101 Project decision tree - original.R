#BT2101 Project decision tree
library(tidyverse)
library(caret)
library('rpart')
library('rpart.plot')
library(mlbench)
library(caret)

data = as_tibble(read.csv('final_data.csv'))
data = data[,3:16]

## 75% of the sample size
smp_size <- floor(0.75 * nrow(data))

## set the seed to make your partition reproducible
set.seed(3000)
train_ind1 <- sample(seq_len(nrow(data)), size = smp_size)
train1 <- data[train_ind1, ]
test1 <- data[-train_ind1, ]

train_ind2 <- sample(seq_len(nrow(data)), size = smp_size)
train2 <- data[train_ind2, ]
test2 <- data[-train_ind2, ]

## all variables (age + sex + cp + trestbps + chol + fbs + restecg + thalach + exang + oldpeak + slope + ca + thal) considered
fit1 <- rpart(target ~ ., data=train1, method="class")
rpart.plot(fit1, extra = 106)
summary(fit1)
prediction1 <- predict(fit1, test1, type = "class")
matrix1 = confusionMatrix(factor(test1$target), factor(prediction1))
matrix1

fit2 <- rpart(target ~ ., data=train2, method="class")
rpart.plot(fit2, extra = 106)
summary(fit2)
prediction2 <- predict(fit2, test2, type = "class")
matrix2 = confusionMatrix(factor(test2$target), factor(prediction2))
matrix2

## testing
control <- trainControl(method="repeatedcv", number=10, repeats=3)
data$target = as.factor(data$target)
# train the LVQ model
set.seed(3000)
modeldt <- train(target ~ ., data=data, method="rpart", trControl=control)
modeldt
