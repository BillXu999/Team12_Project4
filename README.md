---
title: "STA 207 Project 4, Bank Marketing"
output: 
  pdf_document: default
  html_document: 
    df_print: paged
    fig_caption: yes
    number_sections: true
---

<style type="text/css">

body{ /* Normal  */
      font-size: 12px;
  }
math {
  font-size: tiny;
}  
</style>

```{r setup, include=FALSE}
knitr::opts_chunk$set(message=FALSE,warning=FALSE)
```

Team ID: 12

Name (responsibilities): Joseph Gonzalez (Proofread, Introduction, Background)

Name (responsibilities): Yanhao Jin (Logistic Regression, Random Forest, Model Comparison)

Name (responsibilities): Ruichen Xu (Descriptive analysis, Logistic Regression Diagnostics)

Name (responsibilities): Bohao Zou (Logistic Regression Diagnostics)

\newpage

# 1. Introduction

## 1.1 Background

```{r echo=FALSE}
library(C50)
library(descr)
library(caret)
library(randomForest)
library(e1071)
library(xgboost)
library(MLmetrics)
library(ROCR)
```
Businesses rely on data-driven solutions to overcome economic instability and contend with new competitors. These data-driven solutions often reflect customer characteristics and employ data-mining techniques to analyze or predict customer behavior. Using the information obtained from the customers’ data, businesses can strategically plan initiatives to influence attention to their services, maintain their current customer base, and expand their reach to new clients.

From 2008 to 2013, a Portuguese retail bank conducted a direct marketing campaign to persuade new customers to commit to a long term deposit with favorable interest rates. The bank communicated with customers through telephone calls and, during these calls, they documented the customers’ personal characteristics and whether they said “yes” of “no” to signing up for a long term deposit. After this campaign, researchers obtained this customer information dataset from the bank and were interested in constructing a model to explain the bank’s success in obtaining new clients.

Similar to these researchers, we are interested in building a logistic model to predict whether a customer will commit to a long-term deposit. The model we will first use to represent this situation is the logistic regression, which can properly generate the “yes” or “no” binary outcomes for the banking subscription. Our logistic model will be compared to random forest model and xgboost model with cross-validation, to identify differences in performance. From this comparison, we will obtain the information to accurately explain the performance gap and recommend the most business efficient model to our supervisor. Finally, we will provide some practical suggestions for the policymaker to help them conduct next compaign more efficiently. Come up with solutions for the next marketing campaign to expand company’s reach to new clients. We decided to use the bank-additional-full dataset because it has the largest size and most variables for the campaign.

```{r echo=FALSE}
#setwd("C:/Users/yanha/Downloads/bank")
BMdata<- read.table("bank-additional-full.txt",header = TRUE, sep = ";")
### Splitting Data
set.seed(123)
index <- createDataPartition(BMdata$y, p = 0.7, list = FALSE)
set.seed(42)
train_data <- BMdata[index, ]
test_data  <- BMdata[-index, ]
```

## 1.2 Descriptive analysis

```{r echo=FALSE, fig.height=2.5}
library(magrittr)
BankData <- BMdata
TheA<- data.frame(job = BankData$job,month = BankData$month,day_of_week = BankData$day_of_week,contact = BankData$contact,campaign = BankData$campaign, y = BankData$y)
TheA$campaign<- as.factor(TheA$campaign)

library(inspectdf) # To show the overview of data
temp<-inspect_cat(TheA)
BankData_Yes<- TheA[TheA$y == "yes",]
temp<-inspect_cat(BankData_Yes)
show_plot(temp, text_labels = TRUE)
```
Figure 1.2.1: Frequency of categorical levels with subscribing the term deposit

```{r echo=FALSE, fig.height=2.5}
BankData_No<- TheA[TheA$y == "no",]
library(inspectdf) # To show the overview of data
temp<-inspect_cat(BankData_No)
show_plot(temp, text_labels = TRUE)
```
Figure 1.2.2: Frequency of categorical levels without subscribing the term deposit 

In this section, we provide a preliminary description of the factor and numeric variables. We are interested in the distribution of related variables, the percentages for each factor variable and the density map for the numeric variables. Now we can observe the relationship between different levels of factor variables and whether the customers commit to a long term deposit. Figure 1.2.1 and Figure 1.2.2 show that the proportion of people using cellular is larger among those who say "yes" to a long term deposit. This suggests that people using cellular are more inclined to commit to a long term deposit than people using a telephone. The ratio of The day of week is roughly the same whether the the customer accepts or does not accept the long term deposit. With respect to the job variable, blue-collar accounts for more people who say "yes" than people who do not. The proportion of government personnel among those who accept the long term deposit is higher than the proportion who do not accept the long term deposit. In terms of months, the percentage of people who agreed to the deposit in May was significantly larger than those who did not agree to the deposit. According to the Figure 1.2.3, we can inspect the relationship between the number of variables and whether a customer will say "yes" to a long term deposit. In Figure (A), the last contact duration of those who accept the long term deposit offer is significantly longer than those who say "no" to the deposit offer. We can speculate that the longer the communication time, the more likely that people will say "yes."  In Figure (B), the employment variation rate is close to 1 meaning that the density is significantly lower than those who do not want the long term deposit. This implies that the greater the value of employment variation rate, the more likely it is that people will refuse to order.  In Figure (C), we see that people are more willing to refuse the long term deposit when the consumption price index is close to 94.

```{r echo=FALSE, fig.height=4}
library(ggplot2)

plot1<-ggplot(BankData, aes(duration, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()+labs(title = "A: the density of duration")

plot3<-ggplot(BankData, aes(emp.var.rate, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()+labs(title = "B: employment variation rate")
plot4<-ggplot(BankData, aes(cons.price.idx, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()+labs(title = "C: consume price index")

library(ggpubr)
ggarrange(plot1,plot3,plot4,ncol=1,nrow=3)
```
Figure 1.2.3: Density of numerical variables. (A). The density of duration; (B). The density of employment variation rate; (C). The density of consume price index.

# 2. Statistical Analysis

To train classifiers on the bank marketing data set and evaluate the performance of the classifiers, the whole data set was split randomly into a training set ($\small 70\%$ of the whole dataset) and a testing set ($\small 30\%$ of the whole dataset). The proportion of a ”yes” for the response variable in the training/testing set is roughly the same as the whole dataset. When training the classifier, stratified cross-validation is applied to avoid
overfitting. The training set is split into 10 folds. In each fold, the proportion of a ”yes” for the response variable are roughly the same as the whole training data. Finally, we did not do anything on the test data.

The three models we propose for the Bank Marketing Project are:

Logistic Regression Model: We develop logistic regression model by 10-fold cross-validation. The training dataset is splitted into 10 subsets. For each validation, 9 out of 10 subsets are used to fit a logistic regression model, and the remaining one subset is used to calculate the accuracy. The final model would be the best one in 10 fitted logistic regression model with highest accuracy. The logistic regression model is $\small \log\frac{P}{1-P}=\beta_{0}+\sum_{i=1}^{p}\beta_{i} X_{i}$
where $\small P$ is the probability that the client says "yes"(make the subscribution), $\small X_{i}$'s are the selected variables in the logistic regression model, $\small \beta_{i}$ is the coefficient of $\small X_{i}$ and $\small p$ is the number of selected variables. Assumptions of logistic regression model are (1) the response variable to be binary. (2) the observations to be independent of each other. (3) little or no multicollinearity among the independent variables. (4) linearity of independent variables and log odds. (5) the sample size is large enough. 

Random Forest: The random forest model applies the technique of bagging to decision trees. Given a training set $\small \mathbf{X}=\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{n}\}$, where $\small \mathbf{x}_{i}$ is the documented characteristics vector for the $\small i$-th subject in the bank marketing dataset($\small i=1,2,\dots,28831$), the algorithm fits $\small 20$ independent trees to these samples. The output for each decision tree is the probability of the clients saying "yes" to the long term deposit. Then the predictions for new subject $\mathbf{x}^{\prime}$ can be made by averaging the predictions from all decision trees $\small \hat{f}(\mathbf{x}')=\frac{1}{20} \sum_{b=1}^{20} f_{b}\left(\mathbf{x}^{\prime}\right)$. The number of predictors in our model is determined by 10-fold cross-validation. We choose the number of predictors in the final model that maximizes the accuracy. No formal distributional assumption is made for the random forest method. 

Gradient boost tree by xgboost: Given the training dataset $\small \mathbf{X},\mathbf{Y}$, we develop the gradient boost tree by reconstructing the unknown functional dependence $\small \mathbf{X}\rightarrow \mathbf{Y}$ with some estimated model $\hat{f}(\mathbf{X})$, such that the empirical binomial loss function $L=\sum_{i=1}^{28831}\Psi(\mathbf{y}_{i},\hat{f}(\mathbf{x}_{i}))$ is minimized (where $\Psi$ is the empirical binomial loss function for our project). We achieve this goal by the iterative procedure, which starts with a prespecified decision tree. In each step, the procedure builds a new decision tree based on the previous tree to improve the result by minimizing the empirical binomial loss function as much as possible. The procedure stops when the decrease of loss function is less than a specific amount. In R, the `xgboost` package can efficiently develop the gradient boost tree. The assumption for this method is that the binomial loss function’s subgradients are well defined and it is automatically satisfied in our project. [1.][2.] There are two hyperparameters we need to determine by cross validation. They are the max depth of the trees and the colomun sampling ratio by trees.

In the data set, we notice that there are 4640 Yes and 36548 No. Therefore, the dataset is highly imbalanced. We also trained random forest with undersampling methods using `caret` package in R to deal with imbalancedness of the data set. We train the random forest with undersampling methods and oversampling methods.

Finally, to evaluate the performance, the confusion matrix for each classifier above is provided and the Accuarcy, Sensitivity and Kappa value for all classifiers are provided. In particular, we mainly focus on the sensitivity because the sensitivity measures how much percentage of actual subscribers will be predicted to be positive. Thus, higher sensitivity means a greater chance to catch the potential subscribers and sensitivity is more related to banks’ profits. Besides, the kappa value for each classifier is also considered because it measures how closely the instances classified by the classifier matched the data labeled as ground truth. The larger kappa value is larger, the classifer is more reliable in practice.

# 3. Results


## 3.1 Logistic Regression

### 3.1.1 Model chosen by AIC

For interpretability, the stepAIC procedure is applied to reduce the number of variables in logsitic regression. The fitted logistic regression model by 10-fold cross-validation is $$\small \log \frac{P}{1-P}=\beta_{0}+\sum_{i=1}^{9}\beta_{i}X_{i}$$
where $X_{1}$(age), $X_{2}$(marital status), $X_{3}$(education), $X_{4}$(has housing loan), $X_{5}$(last contact duration), $X_{6}$(employment variation rate), $X_{7}$(consumer price index), $X_{8}$(consumer confidence index) and $X_{9}$(euribor 3 month rate) are the selected variables. The coefficients of these variables are shown in Table 5.1.1 in Appendix 5.1. In particular, the log ratio will increase 0.009488 when age increases one unit given other variables fixed. The log ratio will increase 0.004574 if the duration increases one unit given other variables fixed. The log ratio will increase 1.123 when the consumer price index increases one unit given other variables fixed. The log ratio will increase 0.006449 when consumer confidence index increases one unit given other variables fixed. The log ratio will decrease 0.641 if the employment variation rate increases one unit given other variables fixed and the log ratio will decrease 0.3547 when euribor3m variables add one unit given others variables fixed. 


```{r echo=FALSE}
### Logistic Regression
set.seed(42)
control <- trainControl(method = "cv",
                        number = 10,
                        classProbs = TRUE,
                        summaryFunction = multiClassSummary)
model_glm <- train(y~age+marital+education+housing+duration+emp.var.rate+cons.price.idx+cons.conf.idx+euribor3m,data = train_data,                               method = "glm",family = "binomial", trControl = control)
pred_glm_raw <- predict.train(model_glm,
                              newdata = test_data,
                              type = "raw") # use actual predictions
pred_glm_prob <- predict.train(model_glm,
                               newdata = test_data,
                               type = "prob") # use the probabilities
cm_logistic <- confusionMatrix(data = pred_glm_raw,
                factor(test_data$y),
                positive = "yes")

```

The confusion matrix is given by Table 3.1.1.1 The classifier makes a total of 12356 predictions in the test set. Out of these cases, the logistic regression model predicts "yes" 685 times and "no" 11671 times. In reality, 1392 clients actually subscribe the deposit and 10964 clients do not subscribe the client. In particular, there are 436 clients that we predict "yes"(they will subscribe the deposit) and they do subscribe the deposit. There are 10715 clients that we predict "no"(they will not subscribe the deposit) and they do not subscribe the deposit. There are 249 clients that we predict "yes" but actually they do not subscribe the deposit and there are 956 clients that we predict "no" but actually they actually subscribe the deposit. The sensitivity is 0.3132. It measures how often does the random forest predict a client as "yes" when the client actually subscribes the deposit. The specificity is 0.9772. It measures how often does the logistic model predict a client as "no" when the client actually does not subscribe the deposit. Besides, the precision of this logistic model is 0.9205. It measures when the prediction is "yes", how often is it correct. The AUC which measures the goodness of classification, of our random forest is 0.9163817. It is quite close to 1 and thus, the logistic regression model seems to be good.

|               | Actual class: Yes| Actual class: No  |
|---------------|------------------|-------------------|
|Prediction: Yes|436               |249                |
|Prediction: No |956               |10715              |
        Table 3.1.1.1 Confusion Matrix for Logistic Regression by Cross-Validation

Now we check the model assumptions for logistic regression: (1) In the logistic regression model, the assumptions of binary response and large sample size are automatically satisfied. (2) Since the bank communicated with customers through telephone calls independently, the assumption of independence is also roughly satisfied. (3) The VIFs of 5 numeric variables (age, emp.var.rate, cons.price.idx, cons.conf.idx and euribor3m) in the model are calculated to detect the multicolinearity. They are 1.0085, 1.0114, 2.6824, 2.9288, 1.2967 and 2.4287 respectively. These VIF are all less than 10. This indicates there is no strong multicolinearity among those variables. (4) Pearson correlations are calculated to detect the linearity of independent variables and log odds. The Pearson correlation between log odds and those variables are -0.0036(ages), 0.7884(duration), -0.6632(employment variation rate), -0.5324(consumer price index), -0.0684(consumer confidence index) and -0.6448(euribor 3 month rate). The results shows that the variables age and consumer confidence index are not linear with the log odds. These two variables need to be carefully considered in the future analysis. 


### 3.1.2 Full model 
To begin with, the logistic regression model with all variables included in the model is fitted. The model is given by
$$\small \log \frac{P}{1-P}=\beta_{0}+\sum_{i=1}^{21} \beta_{i} X_{i}$$
where $\small P$ is the probability of a client saying "yes" to the company. In particular, the significant coefficient are partially provided in Table 3.3.1

|                          |Etimate   | Std. Error | Pr(>|z|)  |     |
|--------------------------|----------|------------|-----------|-----|
|job(blue-collar)          |-2.724e-01|9.599e-02   |0.004544   |**   |
|job(retired)              |3.789e-01 |1.267e-01   |0.002773   |**   |
|job(student)              |3.716e-01 |1.309e-01   |0.004517   |**   |
|contact(telephone)        |-5.129e-01|9.083e-02   |1.63e-08   |***  |
|month(Aug)                |8.735e-01 |1.446e-01   |1.52e-09   |***  |
|month(June)               |-5.492e-01|1.532e-01   |0.000336   |***  |
|month(Mar)                |1.994e+00 |1.722e-01   |<2e-16     |***  |
|month(May)                |-4.914e-01|9.949e-02   |7.86e-07   |***  |
|month(Nov)                |-4.293e-01|1.445e-01   |0.002970   |**   |
|Day of week(Mon)          |-1.720e-01|7.907e-02   |0.029646   |*    |
|Day of week(Wed)          |1.809e-01 |7.765e-02   |0.019828   |*    |
|duration                  |4.748e-03 |8.947e-05   |< 2e-16    |***  |
|campaign                  |-4.381e-02|1.389e-02   |0.001611   |**   |
|CPI                       |2.094e+00 |3.030e-01   |4.83e-12   |***  |
|Employment variation rate |-1.701e+00|1.707e-01   |< 2e-16    |***  |
Table 3.1.2.1 Significant coefficient for logistic regression with all variables involved in the model.

When the coefficient of the variable is positive, people are more inclined to accept the long term deposit; on the contrary, when the coefficient of the variable is negative, people are more inclined to reject the long term deposit. From the job variable, blue-collar workers more incline to reject the long term deposit; retirees and students are more inclined to accept the long term deposit. If people use telephone to communicate, people are more inclined to reject the long term deposit. From the perspective of the month, people in August And May are more inclined to accept the long term deposit, on the contrary people in June, March, November are more inclined to refuse. From a weekly perspective, it is easier for people to accept the long term deposit on Monday, and people are more inclined to refuse on Wednesday. The longer the duration, the more likely people tend to accept the long term deposit. The larger the campaign, the more likely people tend to reject the long term deposit. The larger the CPI, the more likely people are to accept the long term deposit. The greater the employment variantion rate means People are more likely to refuse the long term deposit.


In particular, if the coefficient of the variable in Table 3.1.2.1 is positive, then this factor will increase the probability of a client saying "yes". If the coefficient of the variable in Table 3.1.1. is negative, then this factor will decrease the probability of a client saying "yes". 

```{r echo=FALSE}
### Logistic Regression
set.seed(42)
control <- trainControl(method = "cv",
                        number = 10,
                        classProbs = TRUE,
                        summaryFunction = multiClassSummary)
model_glm <- train(y~.,data = train_data, method = "glm",family = "binomial", trControl = control)
pred_glm_raw <- predict.train(model_glm,
                              newdata = test_data,
                              type = "raw") # use actual predictions
pred_glm_prob <- predict.train(model_glm,
                               newdata = test_data,
                               type = "prob") # use the probabilities
cm_logistic <- confusionMatrix(data = pred_glm_raw,
                factor(test_data$y),
                positive = "yes")
```

The confusion matrix is given by Table 3.1.2.2 The classifier makes a total of 12356 predictions in the test set. Out of these cases, the logistic regression model predicts "yes" 685 times and "no" 11671 times. In reality, 1392 clients actually subscribe the deposit and 10964 clients do not subscribe the client. In particular, there are 436 clients that we predict "yes"(they will subscribe the deposit) and they do subscribe the deposit. There are 10715 clients that we predict "no"(they will not subscribe the deposit) and they do not subscribe the deposit. There are 249 clients that we predict "yes" but actually they do not subscribe the deposit and there are 956 clients that we predict "no" but actually they actually subscribe the deposit. The accuracy is 0.902. The sensitivity is 0.313. It measures how often does the random forest predict a client as "yes" when the client actually subscribes the deposit. The kappa value for the logistic model is given by 0.373

|               | Actual class: Yes| Actual class: No  |
|---------------|------------------|-------------------|
|Prediction: Yes|436               |249                |
|Prediction: No |956               |10715              |
Table 3.1.2.2 Confusion Matrix for Logistic Regression by Cross-Validation



## 3.2 Random Forest

```{r echo=FALSE}
### Ordinary Random Forest
set.seed(42)
control <- trainControl(method = "cv",
                        number = 10,
                        classProbs = TRUE,
                        summaryFunction = multiClassSummary)
rfGrid <- expand.grid(mtry = seq(from = 4, to = 20, by = 2))
 model_rf <- train(y~.,
                  data = train_data,
                  method = "rf",
                  ntree = 20,
                  tuneLength = 5,
                  trControl = control,
                  tuneGrid = rfGrid)
 pred_rf_raw <- predict.train(model_rf,
                             newdata = test_data,
                             type = "raw")
 pred_rf_prob <- predict.train(model_rf,
                              newdata = test_data,
                              type = "prob")
cm_original <- confusionMatrix(data = pred_rf_raw,
                               factor(test_data$y),
                               positive = "yes")
```

The random forest is one of our alternative approaches in our project. The plot of the accuracy with respect to the number of the predictors is given by Figure 3.3.1 (Top). The number of the predictors in our forest is $\small 10$ with the highest average accuracy $\small 91.25\%$ calculated by cross-validation. 


|               | Actual class: Yes| Actual class: No  |
|---------------|------------------|-------------------|
|Prediction: Yes|668               |369                |
|Prediction: No |724               |10595              |
                 Table 3.2.1 Confusion Matrix for Random Forest Data


The confusion matrix is given by Table 3.2.1. The random forest predicts "yes" 1037 times and "no" 11319 times. In particular, there are 668 clients that we predict "yes" and they do subscribe the deposit. There are 10595 clients that we predict "no" and they do not subscribe the deposit. There are 369 clients that we predict "yes" but actually they do not subscribe the deposit and there are 724 clients that we predict "no" but actually they actually subscribe the deposit. The accuracy of the random forest on original data is 0.911. The sensitivity is 0.479. It measures how often does the random forest predict a client as "yes" when the client actually subscribes the deposit. The kappa value for the random forest on original dataset is 0.502

## 3.3 XG Boost

```{r echo=FALSE}
### XGBoost
# parameter grid for XGBoost
parameterGrid <-  expand.grid(eta = 0.1, # shrinkage (learning rate)
                              colsample_bytree = c(0.5,0.7), # subsample ration of columns
                              max_depth = c(3,6), # max tree depth. model complexity
                              nrounds = 10, # boosting iterations
                              gamma = 1, # minimum loss reduction
                              subsample = 0.7, # ratio of the training instances
                              min_child_weight = 2) # minimum sum of instance weight

model_xgb <- train(y~.,
                   data = train_data,
                   method = "xgbTree",
                   trControl = control,
                   tuneGrid = parameterGrid)

pred_xgb_raw <- predict.train(model_xgb,
                              newdata = test_data,
                              type = "raw")
pred_xgb_prob <- predict.train(model_xgb,
                               newdata = test_data,
                               type = "prob")
cm_xgb<-confusionMatrix(data = pred_xgb_raw,
                factor(test_data$y),
                positive = "yes")
```


The gradient boosting tree by xgboost is our another alternative approach in our project. The plot of the accuracy with respect to the max tree depth for subsample ratio of columns equals to 0.5 and 0.7 is given by Figure 3.3.1 (Bottom). The number of the predictors in our forest is $\small 10$ with the highest average accuracy of the cross validation is $\small 91.25\%$. 

|               | Actual class: Yes| Actual class: No  |
|---------------|------------------|-------------------|
|Prediction: Yes|609               |279                |
|Prediction: No |783               |10685              |
                  Table 3.3.2 Confusion Matrix for Gradient Boosting Tree by XGBoost.

```{r echo=FALSE, fig.height=1.8}
plot(model_rf)
```
```{r echo=FALSE, fig.height=2}
plot(model_xgb)
```
Figure 3.3.1 Top: The plot of the accuracy with respect to the number of the predictors. Bottom: The plot of accuracy of gradient boosting tree model with respect to max tree depth by subsample ratio of columns 0.5 and 0.7.

The confusion matrix is given by Table 3.3.2. The gradient boosting model predicts "yes" 888 times and "no" 11468 times. In particular, there are 609 clients that we predict "yes" and they do subscribe the deposit. There are 10685 clients that we predict "no" and they do not subscribe the deposit. There are 279 clients that we predict "yes" but actually they do not subscribe the deposit and there are 783 clients that we predict "no" but actually they actually subscribe the deposit. The accuracy of the classifier by XGBoost is 0.913. The sensitivity is 0.437 which measures how often does the model predict a client as "yes" when the client actually subscribes the deposit. And the Kappa value of the classifier by XGBoost is 0.510.

## 3.4 Random Forest on the Under-sampled dataset

```{r include=FALSE}
ctrl <- trainControl(method = "repeatedcv", 
                     number = 3, 
                     repeats = 3, 
                     verboseIter = FALSE,
                     sampling = "down")

set.seed(42)
model_rf_under <- caret::train(y ~ .,
                               data = train_data,
                               method = "rf",
                               trControl = ctrl)
final_under <- data.frame(actual = test_data$y,
                          predict(model_rf_under, newdata = test_data, type = "prob"))
final_under$predict <- ifelse(final_under$yes > 0.5, "yes", "no")
final_under$predict <- as.factor(final_under$predict)
cm_under <- confusionMatrix(final_under$predict, test_data$y)
```

Note that the dataset is highly unbalanced. The proposition of "yes" in response is much lower than that of "no". This issue will make the accuracy of our classifier not reliable when we evaluate the performance of the model. Therefore, here we are interested in the performance of the random forest model on the under-sampled data. The data are proprocessed by under-sampling the majority class "no" from our original dataset.

|               | Actual class: Yes| Actual class: No  |
|---------------|------------------|-------------------|
|Prediction: Yes|1276              |1586               |
|Prediction: No |116               |9378               |
            Table 3.4.1 Confusion Matrix for Random Forest in undersampled dataset.

The confusion matrix is given by Table 3.4.1. The undersampled random forest predicts "yes" 2862 times and "no" 9494 times. In particular, there are 1276 clients that we predict "yes" and they do subscribe the deposit. There are 9378 clients that we predict "no" and they do not subscribe the deposit. There are 1586 clients that we predict "yes" but actually they do not subscribe the deposit and there are 116 clients that we predict "no" but actually they actually subscribe the deposit. The accuracy of the random forest model for undersampled data is 0.862. The sensitivity of the random forest in undersampled dataset is 0.916 and the kappa value of this classifier is 0.516.

## 3.5 Random Forest on the oversampled dataset

In this subsection, the random forest model on the over-sampled data is trained. The data are proprocessed by over-sampling the minority class "yes" from our original dataset.

|               | Actual class: Yes| Actual class: No  |
|---------------|------------------|-------------------|
|Prediction: Yes|1297              |1663               |
|Prediction: No |95                |9301               |
            Table 3.5.1 Confusion Matrix for Random Forest in undersampled dataset.

The confusion matrix is given by Table 3.5.1. The oversampled random forest predicts "yes" 2960 times and "no" 9396 times. In particular, there are 1297 clients that we predict "yes" and they do subscribe the deposit. There are 9301 clients that we predict "no" and they do not subscribe the deposit. There are 1663 clients that we predict "yes" but actually they do not subscribe the deposit and there are 95 clients that we predict "no" but actually they actually subscribe the deposit. The accuracy of the random forest model for undersampled data is 0.857. The sensitivity of the random forest in undersampled dataset is 0.931 and the kappa value of this classifier is 0.523. The code for training random forest on oversampled dataset is provided in Appendix 5.4

## 3.6 Model Comparison

We compare above three models (logistic regression, random forest and gradient boosting trees using xgboost) by comparing the sensitivity and kappa value for all classifiers

|Classifier                           |Sensitivity|Kappa|
|-------------------------------------|-----------|-----|
|Logistic Regression                  | 0.313     |0.373|
|Random Forest on Original Data       |0.478      |0.502|
|Classifier by XGBoost                |0.437      |0.510|           
|Random Forest on Under-sampled Data  |0.916      |0.516|
|Random Forest on Over-sampled Data   |0.931      |0.523|
Table 3.6.1. Model comparison based on sensitivity and kappa value.

The random forest models on under/over-sampled data work better in terms of sensitivity. The under/over-sampling methods solve the unbalancedness of original data set quite well. The sensitivity of random forest on under/oversampled data are much larger than other classifiers. And Kappa values for random forest and XGboost classifiers are also larger than that of logistic regression model.

Table 3.6.1. suggests that the random forest models on under/over-sampled data work better in terms of sensitivity and kappa value. When predicting a new client, the random forest model is also more reliable.

Based on the above analysis, we recommend the random forest model on over-sampled data set as our classifier to predict the new client in the next campaign saying "yes" or "no" because this model has largest sensitivity and largest kappa value. In other words, this model has the best performance on catching the true subscribers and reliability.

# 4. Conclusion and Discussion

Based on the descriptive analysis and the results of logistic regression. We provide following solutions for the company.

* First, conduct marketing campaign during the months of March, August and September. Avoid conducting campaign during May, June and November. Clients seem to be more likely to subscribe the deposits especially on March, August and September and more likely to reject the subscription especially May, June and November
* Second, a policy should be implemented that states that no more than 3 calls should be applied to the same potential client in order to save time and effort in getting new potential clients. The more we call the same potential client, the likely he or she will decline to open a term deposit.
* Target potential clients among students or retired person. Potential clients that were students or retired were the most likely to suscribe to a term deposit. Retired individuals, tend to have more term deposits in order to gain some cash through interest payments. Besides, retired individuals and students tend to not spend bigly its cash so they are more likely to put their cash to work by lending it to the financial institution.
* Target individuals with a higher duration: Target the target group that is above average in duration, there is a highly likelihood that this target group would open a term deposit account. This would allow that the success rate of the next marketing campaign would be highly successful.
* Conduct campaign when the economy is increasing or at peak (lower employment variation rate). When the economy is good, people usually have more spare money and thus, they are more likely to subscribe the deposit.

\newpage

# 5. Appendix

## 5.1 Session Information

```{r}
print(sessionInfo(), local = FALSE)
```

## 5.2 Reference

[1.] Friedman, Jerome H. Stochastic gradient boosting. Computational Statistics and Data Analysis, 38(4):367–378, 2002.

[2.] Liaw, Andy and Wiener, Matthew. Classification and regression by random forest. R News, 2(3): 18-22,2002.

[3.] XGBoost: A scalable Tree Boosting System, Tianqi Chen, Carlos Guestrin, ONR (PECASE) N000141010672, NSF IIS 1258741
and the TerraSwarm Research Center sponsored by MARCO and DARPA.


## 5.3 Resources

[1.] https://www.r-bloggers.com/dealing-with-unbalanced-data-in-machine-learning/

[2.] https://rpubs.com/fabiorocha5150/decisiontreemodel?fbclid=IwAR23TCDaBPGzCFVGm7Pf44BQkDdwzHhEIUL-oDut8imL1dT3wIvPdXAcOK0

[3.] https://www.hackerearth.com/zh/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/

[4.] https://www.frontiersin.org/articles/10.3389/fnbot.2013.00021/full

[5.] https://rpubs.com/shienlong/wqd7004_RRookie(Portuguese Bank Marketing Data WQD7004/RRookie/Yong Keh Soon-WQD180065, Vikas Mann-WQD180051, L-ven Lew Teck Wei-WQD180056, Lim Shien Long-WQD180027)

## 5.4 Code for Training Random Forest on Over-sampled Dataset
```{r eval=FALSE}
ctrl2 <- trainControl(method = "repeatedcv", 
                     number = 10, 
                     repeats = 3, 
                     verboseIter = FALSE,
                     sampling = "up")
set.seed(42)
model_rf_over <- caret::train(y ~ .,
                               data = train_data,
                               method = "rf",
                               trControl = ctrl2)
final_over <- data.frame(actual = test_data$y,
                          predict(model_rf_over, newdata = test_data, type = "prob"))
final_over$predict <- ifelse(final_over$yes > 0.5, "yes", "no")
final_over$predict <- as.factor(final_over$predict)
cm_over <- confusionMatrix(final_under$predict, test_data$y)
```

## 5.5 Github information

https://github.com/BillXu999/Team12_Project4/blob/master/README.md

