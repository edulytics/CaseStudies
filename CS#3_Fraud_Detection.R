# Load libraries
library(data.table)
library(ggplot2)
library(plyr)
library(dplyr)
#install.packages('corrplot')
library(corrplot)
library(pROC)
library(glmnet)
library(caret)
#install.packages('Rtsne')
library(Rtsne)
#install.packages('xgboost')
library(xgboost)
#install.packages('doMC')
library(doMC)
#install.packages('randomForest')
library(randomForest)   #For applying Random Forest
library(e1071)          #For SVM
library(rpart)          #For tree models
#install.packages('rpart.plot')
library(rpart.plot)     #for plotting tree
library(caTools)        #For calculating AUC
library(readr)          #Fore reading data
library(caret)

#creditcard <- read.csv("creditcard.csv",header = TRUE)
#psdata <- read.csv("PS_20174392719_1491204439457_log.csv",header = TRUE)
#nrow(psdata)#284807, no missing
library(plyr)
library(tidyverse)
library(caret)
#install.packages('GGally')
library(GGally)
library(stringr)
#install.packages('rattle')
library(rattle)
library(pROC)
#install.packages('ROCR')
library(ROCR)
library(ggplot2)
#install.packages('ggplot2')
#install.packages('colorspace')
#install.packages('ggplot')

#just once --- start
head(psdata)
is_fraud <- psdata %>%
  filter(isFraud == 1)

#head(creditcard)
#summary(creditcard)

not_fraud <- psdata %>%
  filter(isFraud == 0) %>%
  sample_n(12000)

finaldata<- rbind(is_fraud,not_fraud)
nrow(finaldata)
#just once --- end
#write.csv(finaldata,'final_data.csv')

dataf <- read.csv("final_data.csv",header = TRUE)
nrow(dataf)#20213

#pairs(~step+amount+oldbalanceOrg,data=dataf, 
#      main="Simple Scatterplot Matrix")
#abline(lm(mpg~wt), col="red") # regression line (y~x) 
#lines(lowess(wt,mpg), col="blue") # lowess line (x,y)


pairs(plotting_data_frame,diag.panel=panel.hist,
      panel = function(x, y) {
        points(x, y, cex = 0.5)
        abline(lm(y ~ x), lty = "solid", col = "red")
        lines(lowess(x, y))
      }
)

summary(dataf$amount)
sd(dataf$amount)

summary(dataf$newbalanceDest)
sd(dataf$newbalanceDest)
dataf$isFraud <- as.factor(dataf$isFraud)

# ggplot 
ggplot(dataf, aes(x = step, col = isFraud)) + 
  geom_histogram(bins = 743)

#for all variable comparison
require(gdata)
library(readxl)
library(ggplot2)
load("correlation_heat_map.RData")  # from R utility programs
#corrdata <- dataf[,c("step","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest")]
corrdata <- dataf[,c("isFraud","step","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest")]
dataf$step <- as.numeric(dataf$step)
dataf$amount <- as.numeric(dataf$amount)
dataf$oldbalanceOrg <- as.numeric(dataf$oldbalanceOrg)
dataf$newbalanceOrig <- as.numeric(dataf$newbalanceOrig)
dataf$oldbalanceDest <- as.numeric(dataf$oldbalanceDest)
dataf$newbalanceDest <- as.numeric(dataf$newbalanceDest)
correlation_heat_map(corrdata)
ggpairs(corrdata)
summary(dataf$step)
sd(dataf$step)
#pattern of fraud by transaction amount
ggplot(dataf, aes(type, amount, color = isFraud)) +
  geom_point(alpha = 0.01) + 
  geom_jitter()

#hold-out method
#Create a random index. This index will be used to draw our sample.
ind <- sample(2, nrow(dataf), replace=TRUE, prob=c(0.7,0.3))

#Set the training data to where the index = 1 (70%).
#trainData <- dataf[ind==1,]
nrow(trainData) #14107
#write.csv(trainData,"trainData.csv")
trainData <- read.csv("trainData.csv")
#Set the testing data to where the index = 2 (30%)
#testData <- dataf[ind==2,]
#write.csv(testData,"testData.csv")
testData <- read.csv("testData.csv")

nrow(testData) #6106

#for is Fraud
#Find proportion of survivors
x1 <- sum(trainData$isFraud==1)
x2 <- sum(testData$isFraud==1)

trainData$isFraud

summary(trainData)
summary(x2)

#Calculate n for each sample
n1 = nrow(trainData)
n2 = nrow(testData)

#Calculate proportion of survivors for each sample
p1 <- x1/n1
p2 <- x2/n2
#Calculate the total proportion for both samples (p pooled)
p_pooled <- (x1+x2)/(n1+n2)

#Calculate test statistic
zdata <- (p1-p2)/sqrt(p_pooled*(1-p_pooled)*((1/n1)+(1/n2)))
#Calculate p-value
#pvalue<-2*pt(tdata,df=dfs,lower.tail=FALSE)
#dataf$isFraud <- as.numeric(dataf$isFraud)
#trainData$isFraud <- as.numeric(trainData$isFraud)
#testData$isFraud <- as.numeric(testData$isFraud)
str(trainData)

pvalue <- 2*pnorm(-abs(zdata))

zdata; pvalue                    

#for amount
#Calculate sample means
xbar1 <- mean(trainData$amount)
xbar2 <- mean(testData$amount)

#Calculate sample standard deviations
s1 <- sd(trainData$amount)
s2 <- sd(testData$amount)

#Calculate n for each sample
n1 = nrow(trainData)
n2 = nrow(testData)

#Make the degrees of freedom the smaller of the two sample sizes
dfs <- length(testData)

#Calculate the test statistic
tdata<-(xbar1-xbar2)/sqrt((s1^2/n1)+(s2^2/n2))
summary(trainData$amount)
sd(trainData$amount)
summary(testData$amount)
sd(testData$amount)
#Find and display th ep-value
pvalue<-2*pt(tdata,df=dfs,lower.tail=FALSE)
tdata;pvalue
#Chi-Square Test for Homogeneity of Proportions
freq_table1 <- table(ind,dataf$amount)
Xsq_iris1 <- chisq.test(freq_table1)

#for oldbalanceOrg
#Calculate sample means
xbar3 <- mean(trainData$oldbalanceOrg)
xbar4 <- mean(testData$oldbalanceOrg)

#Calculate sample standard deviations
s3 <- sd(trainData$oldbalanceOrg)
s4 <- sd(testData$oldbalanceOrg)

#Make the degrees of freedom the smaller of the two sample sizes
dfs2 <- length(testData)

#Calculate the test statistic
tdata<-(xbar3-xbar4)/sqrt((s3^2/n1)+(s4^2/n2))

#Find and display th ep-value
pvalue<-2*pt(tdata,df=dfs2,lower.tail=FALSE)
tdata;pvalue
summary(testData$oldbalanceOrg)
summary(trainData$oldbalanceOrg)
sd(testData$oldbalanceOrg)
sd(trainData$oldbalanceOrg)

#Chi-Square Test for Homogeneity of Proportions
freq_table2 <- table(ind,dataf$oldbalanceOrg)
Xsq_iris2 <- chisq.test(freq_table2)

#for newbalanceOrig
#Calculate sample means
xbar5 <- mean(trainData$newbalanceOrig)
xbar6 <- mean(testData$newbalanceOrig)

#Calculate sample standard deviations
s5 <- sd(trainData$newbalanceOrig)
s6 <- sd(testData$newbalanceOrig)

#Make the degrees of freedom the smaller of the two sample sizes
dfs3 <- length(testData)

#Calculate the test statistic
tdata3<-(xbar5-xbar6)/sqrt((s5^2/n1)+(s5^2/n2))

#Find and display th ep-value
pvalue3 <- 2*pt(tdata3,df=dfs3,lower.tail=FALSE)
tdata3;pvalue3

summary(testData$newbalanceOrig)
summary(trainData$newbalanceOrig)
sd(testData$newbalanceOrig)
sd(trainData$newbalanceOrig)

#Chi-Square Test for Homogeneity of Proportions
freq_table3 <- table(ind,dataf$newbalanceOrig)
Xsq_iris3 <- chisq.test(freq_table3)

#for oldbalanceDest
#Calculate sample means
xbar7 <- mean(trainData$oldbalanceDest)
xbar8 <- mean(testData$oldbalanceDest)

#Calculate sample standard deviations
s7 <- sd(trainData$oldbalanceDest)
s8 <- sd(testData$oldbalanceDest)

#Make the degrees of freedom the smaller of the two sample sizes
dfs4 <- length(testData)

#Calculate the test statistic
tdata4 <- (xbar7-xbar8)/sqrt((s7^2/n1)+(s8^2/n2))

#Find and display th ep-value
pvalue4 <- 2*pt(tdata4,df=dfs4,lower.tail=FALSE)
tdata4;pvalue4
summary(testData$oldbalanceDest)
summary(trainData$oldbalanceDest)
sd(testData$oldbalanceDest)
sd(trainData$oldbalanceDest)
#Chi-Square Test for Homogeneity of Proportions
freq_table4 <- table(ind,dataf$oldbalanceDest)
Xsq_iris4 <- chisq.test(freq_table4)


#for newbalanceDest
#Calculate sample means
xbar9 <- mean(trainData$newbalanceDest)
xbar10 <- mean(testData$newbalanceDest)

#Calculate sample standard deviations
s9 <- sd(trainData$newbalanceDest)
s10 <- sd(testData$newbalanceDest)

#Make the degrees of freedom the smaller of the two sample sizes
dfs5 <- length(testData)

#Calculate the test statistic
tdata5 <- (xbar9-xbar10)/sqrt((s9^2/n1)+(s10^2/n2))

#Find and display th ep-value
pvalue5 <- 2*pt(tdata5,df=dfs5,lower.tail=FALSE)
tdata5;pvalue5

summary(testData$newbalanceDest)
summary(trainData$newbalanceDest)
sd(testData$newbalanceDest)
sd(trainData$newbalanceDest)

#Chi-Square Test for Homogeneity of Proportions
freq_table5 <- table(ind,dataf$newbalanceDest)
Xsq_iris5 <- chisq.test(freq_table5)

#K Nearest Neighbor Classification
#Install the class package and load the class library
#install.packages("class")
library(class)

#data(iris)
#str(iris)
#Create the k nearest neighbor database using the training data
update_train <- trainData[,c("step","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest",
                             "newbalanceDest")] #,
nrow(update_train)
knn_data <- update_train
#Create the target dataset
#knn_data <-trainData[,c(1:4)]
knn_class <- trainData[,c("isFraud")]

#Create the new data that we want to classify
new_data <- testData[,c("step","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest",
                                 "newbalanceDest")] #,

#Run the k nearest neighbor classifer with k = 3
knn_pred <- knn(knn_data, new_data, knn_class, k=3, prob = TRUE)

#Evaluate the classifier
#First you need to cast the results as vectors
predicted <- as.vector(knn_pred)
actual <- as.vector(testData[,"isFlaggedFraud"])

#Create a Contingency Table
knn_tab <- as.matrix(table(actual=actual, predicted=predicted))

#Create variables needed to compute evaluation metrics 
n <- sum(knn_tab) # number of instances
nc <- nrow(knn_tab) # number of classes
diag <- diag(knn_tab) # number of correctly classified instances per class
rowsums <- apply(knn_tab, 1, sum) # number of instances per class
colsums <- apply(knn_tab, 2, sum) # number of predictions per class
p <- rowsums/n # distribution of instances over the actual classes
q <- colsums/n # distribution of instances over the predicted classes

#Calculate Accuracy
accuracy <- sum(diag)/n

#Calculate per-class precision, recall, and F-1
precision <- diag/colsums
precision <- as.numeric(precision)
recall <- diag/rowsums
recall <- as.numeric(recall)
f1 <- (2*precision*recall)/(precision+recall)
f1[is.nan(f1)] = 0
data.frame(precision, recall, f1)

#Calculate averaged metrics
avgPrecision <- mean(precision)
avgRecall<- mean(recall)
avgF1 <- mean(f1)

##Classification Using Logistic Regression
#Now we'll create a logistic regression model to classify the Virginica species.  You should get a warning.  This means that the data might be linearly separable.  Ignore that for now.
#trainData$newbalanceOrig
model <- glm(isFraud ~ type+amount+oldbalanceOrg+newbalanceOrig+oldbalanceDest+newbalanceDest, family = binomial(logit), data = trainData)
#train_fit<-glm(train_model, family = binomial(link = "logit"), data = traindata)
#Now we'll predict the model 
summary(model)
pr <- predict(model, testData)
#We can now view the model results in a truth table:
table(actual=testData$isFraud, predicted=pr>.5)

z1<-predict.glm(model,newdata=testData,type = "response")
testData$Predict_Fraud <- ifelse((z1 < 0.5), 0, 1)
confusion_matrix <- table(testData$Predict_Fraud, testData$isFraud)
confusionMatrix(confusion_matrix)

library(pROC)
prob=predict(model)
trainData$prob=prob
g <- roc(isFraud ~ prob, data = trainData)
plot(g,col = "blue")  

#Create a Contingency Table
logistic_tab <- as.matrix(table(actual=testData$isFraud, predicted=pr))

#Create variables needed to compute evaluation metrics 
n <- sum(logistic_tab) # number of instances
nc <- nrow(logistic_tab) # number of classes
diag <- diag(logistic_tab) # number of correctly classified instances per class
rowsums <- apply(logistic_tab, 1, sum) # number of instances per class
colsums <- apply(logistic_tab, 2, sum) # number of predictions per class
p <- rowsums/n # distribution of instances over the actual classes
q <- colsums/n # distribution of instances over the predicted classes

#Calculate Accuracy
accuracy3 <- sum(diag)/n

#Calculate per-class precision, recall, and F-1
precision3 <- diag/colsums
recall3 <- diag/rowsums
f3 <- (2*precision3*recall3)/(precision3+recall3)
f3[is.nan(f3)] = 0
data.frame(precision3, recall3, f3)

#Calculate averaged metrics
avgPrecision3 <- mean(precision3)
avgRecall3<- mean(recall3)
avgF3 <- mean(f3)

#Cart classification
#Decision Tree Classification using CART Algorithm
#Install and load the rpart and rpart.plot package
install.packages("rpart")
install.packages("rpart.plot")
library(rpart)
library(rpart.plot)
#model <- glm(isFraud ~ type+amount+oldbalanceOrg+newbalanceOrig+oldbalanceDest+newbalanceDest, family = binomial(logit), data = trainData)
head(trainData)

trainData1 <- trainData[,c("type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","isFraud")]

testData1 <- trainData[,c("type","amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","isFraud")] 
  
rpart_fraud <- rpart(isFraud ~ type+amount+oldbalanceOrg+
                       newbalanceOrig+oldbalanceDest+newbalanceDest, data=trainData1)

print(rpart_fraud)
#Plot the tree
rpart.plot(rpart_fraud)
head(testData1)
NCOL(testData)
#Run the classification model on the test data
rpart_pred <- predict(rpart_fraud, newdata=testData1, type="vector")

#Evaluate the classifier
#First you need to cast the results as vectors
predicted <- as.vector(rpart_pred)
actual <- as.vector(as.vector(testData1[,7]))

#Create a Contingency Table
rpart_tab <- as.matrix(table(actual=actual, predicted=predicted))
write.csv(rpart_tab,"rpart_tab.csv")
#Create variables needed to compute evaluation metrics 
n1 <- sum(rpart_tab) # number of instances
nc1 <- nrow(rpart_tab) # number of classes
diag1 <- diag(rpart_tab) # number of correctly classified instances per class
rowsums1 <- apply(rpart_tab, 1, sum) # number of instances per class
colsums1 <- apply(rpart_tab, 2, sum) # number of predictions per class
p <- rowsums1/n # distribution of instances over the actual classes
q <- colsums1/n # distribution of instances over the predicted classes

#Calculate Accuracy
accuracy1 <- sum(diag)/n

#Calculate per-class precision, recall, and F-1
precision1 <- diag/colsums
recall1 <- diag/rowsums
f2 <- (2*precision1*recall1)/(precision1+recall1)
f2[is.nan(f2)] = 0

data.frame(precision1, recall1, f2)

#Calculate averaged metrics
avgPrecision1 <- mean(precision1)
avgRecall1<- mean(recall1)
avgF2 <- mean(f2)

library(pROC)
prob=predict(rpart_fraud)
trainData1$prob=prob
g <- roc(isFraud ~ prob, data = trainData1)
plot(g,col = "blue")  


