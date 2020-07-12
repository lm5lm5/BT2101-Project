#BT2101 Project

library(mlbench)
library(caret)
library(e1071)
library('readxl')

###SVM
svm_df <- read.csv("final_data.csv", sep = ',', header = TRUE)
svm_df = svm_df[,3:16]
## Data Cleaning
svm_df <- read.csv("final_data.csv", sep = ',', header = TRUE)
svm_df$target[svm_df$target >= 1] <- 1 
str(svm_df)
svm_df$ca[svm_df$ca == "?"] <- NA
svm_df$thal[svm_df$thal == "?"] <- NA
svm_df <- svm_df[complete.cases(svm_df),]
svm_df$ca <- droplevels(svm_df)$ca
svm_df$thal <- droplevels(svm_df)$thal
svm_df$ca <- as.integer(as.character(svm_df$ca))
svm_df$thal <- as.integer(as.character(svm_df$thal))

str(svm_df)
head(svm_df)
summary(svm_df)

# Split the data to training and testing sets
set.seed(3000)
svm_train <- createDataPartition(y = svm_df$target, p= 0.7, list = FALSE)
training <- svm_df[svm_train,]
testing <- svm_df[-svm_train,]

training[["target"]] = factor(training[["target"]])
testing[["target"]] = factor(testing[["target"]])

traincontrol <- trainControl(method = "repeatedcv", number = 10, repeats = 3)

# Case 1: linear form
set.seed(3000)
svm_linear1 <- train(target ~. , data = training, method = "svmLinear", trControl= traincontrol, preProcess = c("center", "scale"), tuneLength = 10)
svm_linear1
test_predict1 <- predict(svm_linear1, newdata = testing)
confusionMatrix(test_predict1, testing$target)

# Case 2: linear form with tuning
grid_linear <- expand.grid(C = c(0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75,
                                 1, 1.5, 2,5))
set.seed(3000)
svm_linear_grid1 <- train(target ~. , data = training, method = "svmLinear", trControl=traincontrol, preProcess = c("center", "scale"),
                         tuneGrid = grid_linear, tuneLength = 10)
svm_linear_grid1
plot(svm_linear_grid1)

test_predict_linear_grid1 <- predict(svm_linear_grid1, newdata = testing)
confusionMatrix(test_predict_linear_grid1, testing$target)

# Case 3: non-linear form

# Radial Kernel
set.seed(3000)
svm_radial1 <- train(target ~., data = training, method = "svmRadial", trControl= traincontrol, preProcess = c("center", "scale"), tuneLength = 10)
svm_radial1
plot(svm_radial1)
test_predict_radial1 <- predict(svm_radial1, newdata = testing)
confusionMatrix(test_predict_radial1, testing$target)

# Case 4: non-linear form with tuning
grid_radial <- expand.grid(sigma = c(0,0.01, 0.02, 0.025, 0.03, 0.04,
                                     0.05, 0.06, 0.07,0.08, 0.09, 0.1, 0.25, 0.5, 0.75,0.9),
                           C = c(0,0.05, 0.1, 0.25, 0.5,
                                 1, 2,5))
set.seed(3000)
svm_radial_grid1 <- train(target ~. , data = training, method = "svmRadial", trControl=traincontrol, preProcess = c("center", "scale"),
                         tuneGrid = grid_radial, tuneLength = 10)
svm_radial_grid1
plot(svm_radial_grid1)

test_predict_radial_grid1 <- predict(svm_radial_grid1, newdata = testing)
confusionMatrix(test_predict_radial_grid1, testing$target)

### glm logistic regression
heart <- read.csv('final_data.csv')
heart = heart[,3:16]

# change to factor
heart$sex <- as.factor(heart$sex)
heart$cp <- as.factor(heart$cp)
heart$fbs <- as.factor(heart$fbs)
heart$restecg <- as.factor(heart$restecg)
heart$exang <- as.factor(heart$exang)
heart$thal <- as.factor(heart$thal)
heart$target <- as.factor(heart$target)

set.seed(3000)
trainIndex <- createDataPartition(heart$target, p = .7, 
                                  list = FALSE, 
                                  times = 1)
heart.train<-heart[trainIndex,]
heart.test<-heart[-trainIndex,]
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  savePredictions = TRUE
)

## Logistic regression
fit1<-train(target ~ age+sex+thalach+exang+oldpeak+ca+thal
           ,data=heart.train,method="glm",family=binomial(),
           trControl=fitControl)
fit1
varImp(fit1)
heart_pred1<-predict(fit1,heart.test)
confusionMatrix(heart_pred1,heart.test$target)

### Naive Bayes
df<- read_csv("final_data.csv")
df<-df[3:16]
df$sex<-unlist(lapply(df$sex,factor))
df$cp<-unlist(lapply(df$cp,factor))
df$fbs<-unlist(lapply(df$fbs,factor))
df$restecg<-unlist(lapply(df$restecg,factor))
df$exang<-unlist(lapply(df$exang,factor))
df$thal<-unlist(lapply(df$thal,factor))
df$target<-unlist(lapply(df$target,as.factor))
set.seed(3000)
rnd <- sample(2, nrow(df), replace = T, prob = c(0.7, 0.3))
df_train <- df[rnd == 1, ]
df_test <- df[rnd == 2, ]

fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 3,
  savePredictions = TRUE
)

set.seed(3000)
fit31 <- train(target ~ ., 
              data=df_train,
              method = "nb",
              trControl = fitControl,
              tuneGrid = data.frame(fL=0, usekernel=FALSE, adjust=F))
pred31 <- predict(fit31, df_test, type="raw")
#Warnings that probability is 0 for some cases
confusionMatrix(pred31, df_test$target)

# collect resamples
results1 <- resamples(list(svm_linear=svm_linear1, svm_linear_grid=svm_linear_grid1, svm_radial_grid=svm_radial_grid1, svm_radial=svm_radial1, logistic_regression=fit1, Naive_Bayes=fit31))
# summarize the distributions
summary(results1)
# boxplots of results
bwplot(results1)
# dot plots of results
dotplot(results1)

fit1_Kappa = fit1$resample$Kappa
fit31_Kappa = na.omit(fit3$resample$Kappa)
svm_linear_grid1_Kappa = svm_linear_grid1$resample$Kappa
svm_radical_grid1_Kappa = svm_radial_grid1$resample$Kappa
var.test(svm_radical_grid1_Kappa, fit31_Kappa)
t.test(svm_radical_grid1_Kappa, fit31_Kappa, var.equal=T)
