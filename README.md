# Team12_Project4

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

Name (responsibilities): Ruichen Xu (Descriptive analysis)

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

Businesses rely on data-driven solutions to overcome economic instability and contend with new competitors. These data-driven solutions often reflect customer characteristics and employ data-mining techniques to analyze or predict customer behavior. Using the information obtained from the customers’ data, businesses can strategically plan initiatives to influence attention to their services, maintain their current customer base, and expand their reach to new clients. From 2008 to 2013, a Portuguese retail bank conducted a direct marketing campaign to persuade new customers to commit to a long term deposit with favorable interest rates. The bank communicated with customers through telephone calls independently and, during these calls, they documented the customers’ personal characteristics and whether they said “yes” of “no” to signing up for a long term deposit. In this project, we are interested in building classification models to predict whether a customer will commit to a long-term deposit. We establish logistic regression model, random forest model and xgboost model for the bank market data, evaluate the performance of the fitted model and then compare all the fitted models.

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
library(inspectdf) # To show the overview of data
temp<-inspect_cat(BankData)
#show_plot(temp, text_labels = TRUE)
BankData_Yes<- BankData[BankData$y == "yes",]
library(inspectdf) # To show the overview of data
temp<-inspect_cat(BankData_Yes)
show_plot(temp, text_labels = TRUE)
```
Figure 1.2.1: Frequency of categorical levels with subscribing the term deposit

```{r echo=FALSE, fig.height=2.5}
BankData_No<- BankData[BankData$y == "no",]
library(inspectdf) # To show the overview of data
temp<-inspect_cat(BankData_No)
show_plot(temp, text_labels = TRUE)
```
Figure 1.2.2: Frequency of categorical levels without subscribing the term deposit 

In this section, we provide a preliminary description of the factor and numeric variables. We are interested in the distribution of related variables, the percentages for each factor variable and the density map for the numeric variables. Now we can observe the relationship between different levels of factor variables and whether the customers commit to a long term deposit. Figure 1.2.1 and Figure 1.2.2 show that the proportion of people using cellular is larger among those who say "yes" to a long term deposit. This suggests that people using cellular are more inclined to commit to a long term deposit than people using a telephone. With respect to the variable "default", the proportion of people who do not have credit is significantly larger than those who say "no" to the deposit. Therefore, we can see that people without credit are more inclined to say "yes" to a long term deposit. As for the education variable, the proportion of people who have a university degree is significantly higher than that of those who say "no". Ultimately, this suggests that those with a university degree are more likely to say "yes". With respect to the job variable, blue-collar accounts for more people who say "yes" than people who do not. The proportion of government personnel among those who accept the long term deposit is higher than the proportion who do not accept the long term deposit. Besides, the proportion of married people who refused the deposit is significantly higher than the proportion of people who agreed to the deposit. In terms of months, the percentage of people who agreed to the deposit in May was significantly larger than those who did not agree to the deposit. With respect to the poutcome variable, people who have previously accepted the deposit plan are more likely to accept another deposit plan. Examining the housing, loan, and day of week variables, we see that the ratio of these factor levels is roughly the same whether the the customer accepts or does not accept the long term deposit. Therefore, the attitude of a customer saying "yes" or "no" to a long term deposit may have nothing to do with housing, loan, and day of the week. According to the Figure 1.2.3, we can inspect the relationship between the number of variables and whether a customer will say "yes" to a long term deposit. In Figure (A), we see that, regardless of whether the customer agrees to a long term deposit, the distribution of age is roughly the same. We can speculate that the year-old collar has no obvious relationship with the decision a customer makes on the deposit offer. In Figure (B), the last contact duration of those who accept the long term deposit offer is significantly longer than those who say "no" to the deposit offer. We can speculate that the longer the communication time, the more likely that people will say "yes."  In Figure (C), the employment variation rate is close to 1 meaning that the density is significantly lower than those who do not want the long term deposit. This implies that the greater the value of employment variation rate, the more likely it is that people will refuse to order.  In Figure (D), we see that people are more willing to refuse the long term deposit when the consumption price index is close to 94. Figure (E) indicates when the confidence index is close to -37, -43, -47, people are more willing to refuse the long term deposit. Figure (F) indicates when euribor 3-month rate is close to 5, people are more prone to say "no" to a long term loan.

```{r echo=FALSE, fig.height=2.8}
library(ggplot2)
plot1<-ggplot(BankData, aes(age, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()+labs(title = "A: the density of age")
plot2<-ggplot(BankData, aes(duration, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()+labs(title = "B: the density of duration")
plot3<-ggplot(BankData, aes(campaign, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()
plot4<-ggplot(BankData, aes(emp.var.rate, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()+labs(title = "C: the density of employment variation rate")
plot5<-ggplot(BankData, aes(cons.price.idx, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()+labs(title = "D: the density of consume price index")
plot6<-ggplot(BankData, aes(cons.conf.idx, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()+labs(title = "E: the density of consume confidence index")
plot7<-ggplot(BankData, aes(euribor3m, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()+labs(title = "F: the density of euribor 3 month rate")

plot8<-ggplot(BankData, aes(nr.employed, fill = y)) +
  geom_density(alpha = 0.5) +
  theme_bw()
library(ggpubr)
ggarrange(plot1,plot2,plot4,plot5,plot6,plot7,ncol=2,nrow=3)
```
Figure 1.2.3: Density of numerical variables. (A). The density of age; (B). The density of duration; (C). The density of employment variation rate; (D). The density of consume price index; (E). The density of consume confidence index; (F). The density of euribor 3 month rate

# 2. Statistical Analysis

To build the classification model, the whole data set was split randomly into a training set ($\small 70\%$ of the whole dataset) and a testing set ($\small 30\%$ of the whole dataset). The three models we propose for the Bank Marketing Project are:

Logistic Regression Model: We develop logistic regression model by 10-fold cross-validation. The training dataset is splitted into 10 subsets. For each validation, 9 out of 10 subsets are used to fit a logistic regression model, and the remaining one subset is used to calculate the accuracy. The final model would be the best one in 10 fitted logistic regression model with highest accuracy. The logistic regression model is $\small \log\frac{P}{1-P}=\beta_{0}+\sum_{i=1}^{p}\beta_{i} X_{i}$
where $\small P$ is the probability that the client says "yes"(make the subscribution), $\small X_{i}$'s are the selected variables in the logistic regression model, $\small \beta_{i}$ is the coefficient of $\small X_{i}$ and $\small p$ is the number of selected variables. Assumptions of logistic regression model are (1) the response variable to be binary. (2) the observations to be independent of each other. (3) little or no multicollinearity among the independent variables. (4) linearity of independent variables and log odds. (5) the sample size is large enough. 

Random Forest: The random forest model applies the technique of bagging to decision trees. Given a training set $\small \mathbf{X}=\{\mathbf{x}_{1}, \ldots, \mathbf{x}_{n}\}$, where $\small \mathbf{x}_{i}$ is the documented characteristics vector for the $\small i$-th subject in the bank marketing dataset($\small i=1,2,\dots,28831$), the algorithm fits $\small 20$ independent trees to these samples. The output for each decision tree is the probability of the clients saying "yes" to the long term deposit. Then the predictions for new subject $\mathbf{x}^{\prime}$ can be made by averaging the predictions from all decision trees $\small \hat{f}(\mathbf{x}')=\frac{1}{20} \sum_{b=1}^{20} f_{b}\left(\mathbf{x}^{\prime}\right)$. The number of predictors in our model is determined by 10-fold cross-validation. We choose the number of predictors in the final model that maximizes the accuracy. No formal distributional assumption is made for the random forest method. 

Gradient boost tree by xgboost: Given the training dataset $\small \mathbf{X},\mathbf{Y}$, we develop the gradient boost tree by reconstructing the unknown functional dependence $\small \mathbf{X}\rightarrow \mathbf{Y}$ with some estimated model $\hat{f}(\mathbf{X})$, such that the empirical binomial loss function $L=\sum_{i=1}^{28831}\Psi(\mathbf{y}_{i},\hat{f}(\mathbf{x}_{i}))$ is minimized (where $\Psi$ is the empirical binomial loss function for our project). We achieve this goal by the iterative procedure, which starts with a prespecified decision tree. In each step, the procedure builds a new decision tree based on the previous tree to improve the result by minimizing the empirical binomial loss function as much as possible. The procedure stops when the decrease of loss function is less than a specific amount. In R, the `xgboost` package can efficiently develop the gradient boost tree. The assumption for this method is that the binomial loss function’s subgradients are well defined and it is automatically satisfied in our project. [1.][2.] There are two hyperparameters we need to determine by cross validation. They are the max depth of the trees and the colomun sampling ratio by trees.

# 3. Results

## 3.1 Logistic Regression

For interpretability, the stepAIC procedure is applied to reduce the number of variables in logsitic regression. The fitted logistic regression model by 10-fold cross-validation is $\small \log \frac{P}{1-P}=\beta_{0}+\sum_{i=1}^{9}\beta_{i}X_{i}$
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

The confusion matrix is given by Table 3.1.1. The classifier makes a total of 12356 predictions in the test set. Out of these cases, the logistic regression model predicts "yes" 685 times and "no" 11671 times. In reality, 1392 clients actually subscribe the deposit and 10964 clients do not subscribe the client. In particular, there are 436 clients that we predict "yes"(they will subscribe the deposit) and they do subscribe the deposit. There are 10715 clients that we predict "no"(they will not subscribe the deposit) and they do not subscribe the deposit. There are 249 clients that we predict "yes" but actually they do not subscribe the deposit and there are 956 clients that we predict "no" but actually they actually subscribe the deposit. The sensitivity is 0.3132. It measures how often does the random forest predict a client as "yes" when the client actually subscribes the deposit. The specificity is 0.9772. It measures how often does the logistic model predict a client as "no" when the client actually does not subscribe the deposit. Besides, the precision of this logistic model is 0.9205. It measures when the prediction is "yes", how often is it correct. The AUC which measures the goodness of classification, of our random forest is 0.9163817. It is quite close to 1 and thus, the logistic regression model seems to be good.

|               | Actual class: Yes| Actual class: No  |
|---------------|------------------|-------------------|
|Prediction: Yes|436               |249                |
|Prediction: No |956               |10715              |
        Table 3.1.1 Confusion Matrix for Logistic Regression by Cross-Validation

Now we check the model assumptions for logistic regression: (1) In the logistic regression model, the assumptions of binary response and large sample size are automatically satisfied. (2) Since the bank communicated with customers through telephone calls independently, the assumption of independence is also roughly satisfied. (3) The VIFs of 5 numeric variables (age, emp.var.rate, cons.price.idx, cons.conf.idx and euribor3m) in the model are calculated to detect the multicolinearity. They are 1.0085, 1.0114, 2.6824, 2.9288, 1.2967 and 2.4287 respectively. These VIF are all less than 10. This indicates there is no strong multicolinearity among those variables. (4) Pearson correlations are calculated to detect the linearity of independent variables and log odds. The Pearson correlation between log odds and those variables are -0.0036(ages), 0.7884(duration), -0.6632(employment variation rate), -0.5324(consumer price index), -0.0684(consumer confidence index) and -0.6448(euribor 3 month rate). The results shows that the variables age and consumer confidence index are not linear with the log odds. These two variables need to be carefully considered in the future analysis. 

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

The random forest is one of our alternative approaches in our project. The plot of the accuracy with respect to the number of the predictors is given by Figure 3.3.1 (Top). The number of the predictors in our forest is $\small 10$ with the highest average accuracy $\small 91.25\%$. 


|               | Actual class: Yes| Actual class: No  |
|---------------|------------------|-------------------|
|Prediction: Yes|668               |369                |
|Prediction: No |724               |10595              |
                 Table 3.2.1 Confusion Matrix for Random Forest Data


The confusion matrix is given by Table 3.2.1. The random forest predicts "yes" 1037 times and "no" 11319 times. In particular, there are 668 clients that we predict "yes" and they do subscribe the deposit. There are 10595 clients that we predict "no" and they do not subscribe the deposit. There are 369 clients that we predict "yes" but actually they do not subscribe the deposit and there are 724 clients that we predict "no" but actually they actually subscribe the deposit. The sensitivity is 0.4989. It measures how often does the random forest predict a client as "yes" when the client actually subscribes the deposit. The specificity is 0.9663. It measures how often does the random forest predict a client as "no" when the client actually does not subscribe the deposit. Besides, the precision of this random forest is 0.9376. It measures when the prediction is "yes", how often is it correct. The AUC which measures the goodness of classification, of our random forest is 0.9311. It is quite close to 1 and thus, the random forest model seems to be good.

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

The confusion matrix is given by Table 3.3.2. The gradient boosting model predicts "yes" 888 times and "no" 11468 times. In particular, there are 609 clients that we predict "yes" and they do subscribe the deposit. There are 10685 clients that we predict "no" and they do not subscribe the deposit. There are 279 clients that we predict "yes" but actually they do not subscribe the deposit and there are 783 clients that we predict "no" but actually they actually subscribe the deposit. The sensitivity is 0.4375 which measures how often does the model predict a client as "yes" when the client actually subscribes the deposit. The specificity is 0.9746. It measures how often does the model predict a client as "no" when the client actually does not subscribe the deposit. Besides, the precision of this model is 0.9269. It measures when the prediction is "yes", how often is it correct. The AUC which measures of the gradient boosting trees is 0.9416. It is quite close to 1 and thus, the gradient boosting tree model seems to be good.

## 3.4 Model Comparison

We compare above three models (logistic regression, random forest and gradient boosting trees using xgboost) by generating the boxplot of key features of classifiers by these methods. The boxplot shown in Figure 3.5.1 is based on the resampling results. This methods make it appropriate to compare the different machine learning methods.

```{r echo=FALSE}
### Model comparison
models <- list(logistic_regression = model_glm,
               randomforest = model_rf,
               xgboost = model_xgb)

resampling <- resamples(models)
```

```{r echo=FALSE, fig.height=1.8}
bwplot(resampling, metric = c("Accuracy", "AUC", "Kappa", "Precision", "Sensitivity", "Specificity"))
```
Figure 3.4.1 multi-boxplot of the key features("Accuracy", "AUC", "F1", "Kappa", "Precision", "Sensitivity", "Specificity") by three methods.

Figure 3.4.1 shows that there seems to be no big difference between three models based on Precision, Sensitivity and Accuracy. The specificity and kappa value is largest in the random forest model and smallest in the logistic regression. The specificity of these three models indicates the random forest model has the largest frequency that the random forest predict a client as "no" when the client actually does not subscribe the deposit. Besides, the kappa value of the classifier measures a metric that compares an Observed Accuracy with an Expected Accuracy. It measures how closely the instances classified by the classifier matched the data labeled as ground truth. The medians of kappa statistics for three models are given by $\small \kappa_{1}=0.3883, \kappa_{2}=0.5105,\kappa_{3}=0.4463$, where $\small \kappa_{1},\kappa_{2}$ and $\small \kappa_{3}$ is the median kappa values for logsitic regression, random forest and xgboost method respectively. Based on the kappa statistic, the random forest model is preferred.

# 4. Conclusion and Discussion

Note that the dataset is highly unbalanced. The proposition of "yes" in response is much lower than that of "no". This issue will reduce the accuracy of our classifier. Therefore, here we are interested in the performance of the random forest model on the under-sampled data. The data are proprocessed by under-sampling the majority class "no" from our original dataset.


|               | Actual class: Yes| Actual class: No  |
|---------------|------------------|-------------------|
|Prediction: Yes|1276              |1586               |
|Prediction: No |116               |9378               |
                Table 4.1 Confusion Matrix for Random Forest in undersampled dataset.

The confusion matrix is given by Table 4.1. The undersampled random forest predicts "yes" 2862 times and "no" 9494 times. In particular, there are 1276 clients that we predict "yes" and they do subscribe the deposit. There are 9378 clients that we predict "no" and they do not subscribe the deposit. There are 1586 clients that we predict "yes" but actually they do not subscribe the deposit and there are 116 clients that we predict "no" but actually they actually subscribe the deposit. The accuracy of the random forest model for undersampled data is 0.8623. Thus, random forest on original data and gradient boosting tree models are better than this model in terms of accuracy. This make sense because in undersampling process some information in the full dataset is lost. Besides, due to the computation complexity of random forest on over-sampled dataset, we did not train the corresponding classifier. Oversampling on original dataset might solve the unbalance of the dataset without loosing the information. Besides, the model diagnostic of logistic regression indicates that age and consumer confidence index should be carefully considered when we fit other logistic regression classifiers using these two variables.

\newpage

# 5. Appendix

## 5.1 Detailed Coefficients for Logistic Regression Model

| Coefficients:                 | Estimate | Std. Error | z value | Pr(>\|z\|)|     |
|-------------------------------|----------|------------|---------|-----------|-----|
| (Intercept)                   | -105.9   | 6.531      | -16.218 | < 2*e-16  | *** |
| age                           | 0.0095   | 0.0023     | 4.203   | 2.63*e-05 | *** |
| marital-married               | 0.0034   | 0.0783     | 0.044   | 0.96524   |     |
| marital-single                | 0.2431   | 0.0884     | 2.750   | 0.00596   | **  |
| marital-unknown               | 0.3506   | 0.4234     | 0.828   | 0.40770   |     |
| education-basic.6y            | -0.1263  | 0.1398     | -0.903  | 0.36628   |     |
| education-basic.9y            | -0.1066  | 0.1052     | -1.014  | 0.31076   |     |
| education-high.school         | 0.1074   | 0.0939     | 1.144   | 0.25256   |     |
| education-illiterate          | 1.377    | 0.977      | 1.408   | 0.15902   |     |
| education-professional.course | 0.2731   | 0.1018     | 2.684   | 0.00727   | **  |
| education-university.degree   | 0.4099   | 0.08925    | 4.593   | 4.37*e-06 | *** |
| education-unknown             | 0.2612   | 0.1275     | 2.048   | 0.04051   | *   |
| housing-unknown               | 0.2336   | 0.1588     | -1.471  | 0.14127   |     |
| housing-yes                   | 0.004    | 0.04687    | -0.086  | 0.93116   |     |
| duration                      | 0.00458  | 0.00009    | 53.661  | < 2*e-16  | *** |
| emp.var.rate                  | -0.641   | -0.0769    | -8.333  | < 2*e-16  | *** |
| cons.price.idx                | 1.123    | 0.06787    | 16.543  | < 2*e-16  | *** |
| cons.conf.idx                 | 0.06449  | 0.004576   | 14.122  | < 2*e-16  | *** |
| euribor3m                     | -0.3547  | 0.05836    | -6.077  | 1.22*e-09 | *** |
Table 5.1.1 The Coefficients of Logistic Regression Model

## 5.2 Session Information

```{r}
print(sessionInfo(), local = FALSE)
```

## 5.3 Reference

[1.] Friedman, Jerome H. Stochastic gradient boosting. Computational Statistics and Data Analysis, 38(4):367–378, 2002.

[2.] Liaw, Andy and Wiener, Matthew. Classification and regression by random forest. R News, 2(3): 18-22,2002.

[3.] XGBoost: A scalable Tree Boosting System, Tianqi Chen, Carlos Guestrin, ONR (PECASE) N000141010672, NSF IIS 1258741
and the TerraSwarm Research Center sponsored by MARCO and DARPA.


## 5.4 Resources

[1.] https://www.r-bloggers.com/dealing-with-unbalanced-data-in-machine-learning/

[2.] https://rpubs.com/fabiorocha5150/decisiontreemodel?fbclid=IwAR23TCDaBPGzCFVGm7Pf44BQkDdwzHhEIUL-oDut8imL1dT3wIvPdXAcOK0

[3.] https://www.hackerearth.com/zh/practice/machine-learning/machine-learning-algorithms/beginners-tutorial-on-xgboost-parameter-tuning-r/tutorial/

[4.] https://www.frontiersin.org/articles/10.3389/fnbot.2013.00021/full

[5.] https://rpubs.com/shienlong/wqd7004_RRookie(Portuguese Bank Marketing Data WQD7004/RRookie/Yong Keh Soon-WQD180065, Vikas Mann-WQD180051, L-ven Lew Teck Wei-WQD180056, Lim Shien Long-WQD180027)

## 5.4 Github information

https://github.com/BillXu999/Team12_Project4/edit/master/README.md
