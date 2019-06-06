# Loan Default Prediction 
#loading R modlules 
pacman::p_load("tidyverse","textmineR","qdap","hunspell","CRAN","tm",'ngram',
               "SnowballC","rpart.plot","psych","stringr","rpart","sqldf","e1071",
               "gridExtra","aCRM", "caret", "coefplot", "cluster", "factoextra", 
               "readxl","dplyr","lubridate", "factoextra","geosphere", "textstem", 
               "wordcloud", "doSNOW", "xgboost", "randomForest")

## Loading the CSV file form the loacal machines
df<-read.csv("File Location")

## Removing the unwanted columns from the data 
df<-df[,-c(unwanted columns)]

## Replacing the missing values with suitable values
df$age<-ifelse(is.na(df$age),0, df$age)

## Initiating the age availability indicator
df$age_avilable<-ifelse(df$age==0,0,1)

## 
df$avg_word_len<-ifelse(is.na(df$avg_word_len),0,df$avg_word_len)
df$status<-as.factor(df$status)


set.seed(42)
inTrain <- createDataPartition(df$status, p = 0.7, list = FALSE)


df_train <- df[inTrain,]
df_test <- df[-inTrain,]
prop.table(table(df_train$status))
prop.table(table(df_test$status))
df_train_nontext<-df_train[,1:20]
df_train_text<-df_train[,-c(2:20)]
df_test_nontext<-df_test[,1:20]
df_test_text<-df_test[,-c(2:20)]




##defining functions for accuracy, total cost
Accuracy<-function(predictions){
  conf<-confusionMatrix(predictions,df_test$status)
  return(round(conf$overall['Accuracy'],4)) 
}
total_cost<- function(predictions){
  return(sum(df_test$loan_amount[which(predictions==0 & df_test$status==1)],
             df_test$loan_amount[which(predictions==1 & df_test$status==0)]*.05))
}

TC=data.frame()
AC=data.frame()

# Tree for only text data 
tree_cont<-trainControl(method="repeatedcv", number=10, repeats=5)
tree2<-train(status~.,data=df_train_text, method="rpart",tuneLength=5, trControl=tree_cont)
plot(tree2)
tree2
tree_text<-predict(tree2, df_test_text, type="raw")
confusionMatrix(tree_text,df_test_text$status)
rpart.plot(rpart(status~.,data=df_train_text,cp=tree2$results$cp[tree2$results$Accuracy==max(tree2$results$Accuracy)]), type=3, roundint = F, cex=0.8)

AC[1,1]=Accuracy(tree_text)
TC[1,1]=total_cost(tree_text)



# Tree for not text data 
tree_cont<-trainControl(method="repeatedcv", number=10, repeats=5)
tree1<-train(status~.,data=df_train_nontext, method="rpart",tuneLength=5, trControl=tree_cont)
plot(tree1)
tree1
tree_nontext<-predict(tree1, df_test_nontext, type="raw")
confusionMatrix(tree_nontext,df_test_nontext$status)
rpart.plot(rpart(status~.,data=tree1$results$cp[tree1$results$Accuracy==max(tree1$results$Accuracy)]), type=5, roundint = F, cex=0.55)

AC[1,2]=Accuracy(tree_nontext)
TC[1,2]=total_cost(tree_nontext)





# Tree for complete data 
tree_cont<-trainControl(method="repeatedcv", number=10, repeats=5)
tree3<-train(status~.,data=df_train, method="rpart",tuneLength=5, trControl=tree_cont)
plot(tree3)
tree3
tree<-predict(tree3, df_test, type="raw")
confusionMatrix(tree,df_test$status)
rpart.plot(rpart(status~.,data=tree3$results$cp[tree3$results$Accuracy==max(tree3$results$Accuracy)]), type=5, roundint = F, cex=0.55)

AC[1,3]=Accuracy(tree)
TC[1,3]=total_cost(tree)

### Random Forest on text data
rf_cont<-trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid<-expand.grid(mtry=1:10)
set.seed(1000)
rf_gridsearch<-train(status~.,data=df_train_text,method="rf", tuneGrid=tunegrid, trControl=rf_cont)
rf_gridsearch
plot(rf_gridsearch)
mtry=rf_gridsearch$bestTune

plot(varImp(rf_gridsearch))
varImpPlot(rf_gridsearch,type=2)
rf2<-predict(rf_gridsearch, df_test_text, type="raw")
confusionMatrix(rf2,df_test_text$status)

AC[2,1]=Accuracy(rf2)
TC[2,1]=total_cost(rf2)



### Random Forest on not text data
rf_cont<-trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid<-expand.grid(mtry=1:10)
set.seed(1000)
rf_gridsearch<-train(status~.,data=df_train_nontext,method="rf", tuneGrid=tunegrid, trControl=rf_cont)
rf_gridsearch
plot(rf_gridsearch)
mtry=rf_gridsearch$bestTune

plot(varImp(rf_gridsearch))
varImpPlot(rf_gridsearch,type=2)
rf1<-predict(rf_gridsearch, df_test_nontext, type="raw")
confusionMatrix(rf1,df_test_nontext$status)

AC[2,2]=Accuracy(rf1)
TC[2,2]=total_cost(rf1)






### Random Forest on complete data
rf_cont<-trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid<-expand.grid(mtry=1:10)
set.seed(1000)
rf_gridsearch<-train(status~.,data=df_train,method="rf", tuneGrid=tunegrid, trControl=rf_cont)
rf_gridsearch
plot(rf_gridsearch)
mtry=rf_gridsearch$bestTune

plot(varImp(rf_gridsearch))

rf3<-predict(rf_gridsearch, df_test, type="raw")
confusionMatrix(rf3,df_test$status)

AC[2,3]=Accuracy(rf3)
TC[2,3]=total_cost(rf3)

## XG Boost on text only data 
#using one hot encoding 
new_train2 <- model.matrix(~.+0,data = df_train_text[,-1]) 
new_test2 <- model.matrix(~.+0,data = df_test_text[,-1])

#convert factor to numeric 
train2_label <- as.numeric(df_train$status)-1
test2_label <- as.numeric(df_test$status)-1


dtrain2 <- xgb.DMatrix(data = new_train2,label =train_label) 
dtest2 <- xgb.DMatrix(data = new_test2,label=test_label)

best_param <- list()
best_seednumber <- 1234
best_rmse <- Inf
best_rmse_index <- 0

set.seed(123)
for (iter in 1:100) {
  param <- list(objective = "binary:logistic",
                eval_metric = "rmse",
                max_depth = sample(6:10, 1),
                eta = runif(1, .01, .3), # Learning rate, default: 0.3
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  cv.nround <-  1000
  cv.nfold <-  5 # 5-fold cross-validation
  seed.number  <-  sample.int(10000, 1) # set seed for the cv
  set.seed(seed.number)
  mdcv <- xgb.cv(data = dtrain2, params = param,  
                 nfold = cv.nfold, nrounds = cv.nround,
                 verbose = F, early_stopping_rounds = 8, maximize = FALSE)
  
  min_rmse_index  <-  mdcv$best_iteration
  min_rmse <-  mdcv$evaluation_log[min_rmse_index]$test_rmse_mean
  
  if (min_rmse < best_rmse) {
    best_rmse <- min_rmse
    best_rmse_index <- min_rmse_index
    best_seednumber <- seed.number
    best_param <- param
  }
}

# The best index (min_rmse_index) is the best "nround" in the model
nround = best_rmse_index
set.seed(best_seednumber)
xg_mod2 <- xgboost(data = dtrain2, params = best_param, nround = nround, verbose = F)

xgbpred2 <- predict (xg_mod2,dtest2)
xgbpred2 <- ifelse (xgbpred2 > 0.5,1,0)
confusionMatrix (as.factor(xgbpred2), df_test_text$status)

AC[3,1]=Accuracy(as.factor(xgbpred2))

mat2 <- xgb.importance (feature_names = colnames(dtrain2),model = xg_mod2)

xgb.plot.importance (importance_matrix = mat2[1:20]) 

TC[3,1]=total_cost(xgbpred2)







## XG Boost on nontext data 
#using one hot encoding 
new_train1 <- model.matrix(~.+0,data = df_train_nontext[,-1]) 
new_test1 <- model.matrix(~.+0,data = df_test_nontext[,-1])

#convert factor to numeric 
train1_label <- as.numeric(df_train$status)-1
test1_label <- as.numeric(df_test$status)-1


dtrain1 <- xgb.DMatrix(data = new_train1,label =train_label) 
dtest1 <- xgb.DMatrix(data = new_test1,label=test_label)

best_param <- list()
best_seednumber <- 1234
best_rmse <- Inf
best_rmse_index <- 0

set.seed(123)
for (iter in 1:100) {
  param <- list(objective = "binary:logistic",
                eval_metric = "rmse",
                max_depth = sample(6:10, 1),
                eta = runif(1, .01, .3), # Learning rate, default: 0.3
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  cv.nround <-  1000
  cv.nfold <-  5 # 5-fold cross-validation
  seed.number  <-  sample.int(10000, 1) # set seed for the cv
  set.seed(seed.number)
  mdcv <- xgb.cv(data = dtrain1, params = param,  
                 nfold = cv.nfold, nrounds = cv.nround,
                 verbose = F, early_stopping_rounds = 8, maximize = FALSE)
  
  min_rmse_index  <-  mdcv$best_iteration
  min_rmse <-  mdcv$evaluation_log[min_rmse_index]$test_rmse_mean
  
  if (min_rmse < best_rmse) {
    best_rmse <- min_rmse
    best_rmse_index <- min_rmse_index
    best_seednumber <- seed.number
    best_param <- param
  }
}

# The best index (min_rmse_index) is the best "nround" in the model
nround = best_rmse_index
set.seed(best_seednumber)
xg_mod1 <- xgboost(data = dtrain1, params = best_param, nround = nround, verbose = F)

xgbpred1 <- predict (xg_mod1,dtest1)
xgbpred1 <- ifelse (xgbpred1 > 0.5,1,0)
confusionMatrix (as.factor(xgbpred1), df_test_nontext$status)

AC[3,2]=Accuracy(as.factor(xgbpred1))

mat1 <- xgb.importance (feature_names = colnames(dtrain1),model = xg_mod1)

xgb.plot.importance (importance_matrix = mat1[1:20]) 

TC[3,2]=total_cost(xgbpred1)








## XG Boost on complete data 
#using one hot encoding 
new_train <- model.matrix(~.+0,data = df_train[,-1]) 
new_test <- model.matrix(~.+0,data = df_test[,-1])

#convert factor to numeric 
train_label <- as.numeric(df_train$status)-1
test_label <- as.numeric(df_test$status)-1


dtrain <- xgb.DMatrix(data = new_train,label =train_label) 
dtest <- xgb.DMatrix(data = new_test,label=test_label)

best_param <- list()
best_seednumber <- 1234
best_rmse <- Inf
best_rmse_index <- 0

set.seed(123)
for (iter in 1:100) {
  param <- list(objective = "binary:logistic",
                eval_metric = "rmse",
                max_depth = sample(6:10, 1),
                eta = runif(1, .01, .3), # Learning rate, default: 0.3
                subsample = runif(1, .6, .9),
                colsample_bytree = runif(1, .5, .8), 
                min_child_weight = sample(1:40, 1),
                max_delta_step = sample(1:10, 1)
  )
  cv.nround <-  1000
  cv.nfold <-  5 # 5-fold cross-validation
  seed.number  <-  sample.int(10000, 1) # set seed for the cv
  set.seed(seed.number)
  mdcv <- xgb.cv(data = dtrain, params = param,  
                 nfold = cv.nfold, nrounds = cv.nround,
                 verbose = F, early_stopping_rounds = 8, maximize = FALSE)
  
  min_rmse_index  <-  mdcv$best_iteration
  min_rmse <-  mdcv$evaluation_log[min_rmse_index]$test_rmse_mean
  
  if (min_rmse < best_rmse) {
    best_rmse <- min_rmse
    best_rmse_index <- min_rmse_index
    best_seednumber <- seed.number
    best_param <- param
  }
}

# The best index (min_rmse_index) is the best "nround" in the model
nround = best_rmse_index
set.seed(best_seednumber)
xg_mod <- xgboost(data = dtrain, params = best_param, nround = nround, verbose = F)

xgbpred <- predict (xg_mod,dtest)
xgbpred <- ifelse (xgbpred > 0.5,1,0)
confusionMatrix (as.factor(xgbpred), df_test$status)

AC[3,3]=Accuracy(as.factor(xgbpred))

mat <- xgb.importance (feature_names = colnames(dtrain),model = xg_mod)

xgb.plot.importance (importance_matrix = mat[1:20]) 

TC[3,3]<-total_cost(xgbpred)


#Fitting the Naive Bayes model on text data
Naive_Bayes_Model_text=naiveBayes(status~., data=df_train_text)
#What does the model say? Print the model summary
Naive_Bayes_Model_text

#Prediction on the dataset
NB_Predictions_text=predict(Naive_Bayes_Model_text, df_test_text)
#Confusion matrix to check accuracy
confusionMatrix(NB_Predictions_text,df_test_text$status)
AC[4,1]<-Accuracy(NB_Predictions_text)
TC[4,1]<-total_cost(NB_Predictions_text)


#Fitting the Naive Bayes model on nontext data
Naive_Bayes_Model_nontext=naiveBayes(status~., data=df_train_nontext)
#What does the model say? Print the model summary
Naive_Bayes_Model_nontext

#Prediction on the dataset
NB_Predictions_nontext=predict(Naive_Bayes_Model_nontext, df_test_nontext)
#Confusion matrix to check accuracy
confusionMatrix(NB_Predictions_nontext,df_test_nontext$status)
AC[4,2]<-Accuracy(NB_Predictions_nontext)
TC[4,2]<-total_cost(NB_Predictions_nontext)



#Fitting the Naive Bayes model on complete data
Naive_Bayes_Model=naiveBayes(status~., data=df_train)
#What does the model say? Print the model summary
Naive_Bayes_Model

#Prediction on the dataset
NB_Predictions=predict(Naive_Bayes_Model, df_test)
#Confusion matrix to check accuracy
confusionMatrix(NB_Predictions,df_test$status)
AC[4,3]<-Accuracy(NB_Predictions)
TC[4,3]<-total_cost(NB_Predictions)

mean(df$loan_amount[df$status==1])
mean(df$loan_amount[df$status==0])*.1

AC
TC
