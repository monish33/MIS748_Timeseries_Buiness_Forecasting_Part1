```{r setup, include=FALSE, echo=FALSE, results='asis'}
knitr::opts_chunk$set(echo = TRUE)
# Install the glmnet package if not already installed
if (!require(glmnet, quietly = TRUE)) {
  install.packages("glmnet")
}
library(knitr)
library(kableExtra)

# Load the glmnet package
library(glmnet)
# Load the necessary libraries
library(lubridate)
library(dplyr)
library(ggplot2)
library(tidyverse)
library(caret)
library(MLmetrics)
library(randomForest)
library(xgboost)
library(e1071)
```


## Overview

This project carried out a simple Exploratory data analysis (EDA) analysis on the superstore data set. EDA is primarily used to see what data can reveal beyond the formal modeling and provides a provides a better understanding of data set variables and the relationships between them. In addition to the mentioned analysis, five different regression models were developed and the best fit one recommended.The developed models include Linear Regression,
Polynomial Regression, Ridge Regression, Lasso Regression, and Random Forest Regression.

## Data Source and Data Preparation (Part I)

### Data Source

Superstore data is used in this assignment. The data is retrieved from kaggle. The Dataset contains the information related to sales, Profits and other interesting facts of a Superstore giant.

### Data Preparation

In this section, data wrangling/preparation is carried out. By definition, data preparation is the process of preparing raw data so that it is suitable for further processing and analysis. Key steps include collecting, cleaning, and labeling raw data into a form suitable for modeling/regression algorithms and then exploring and visualizing the data.

#### Loading data
```{r, echo=FALSE}
df <- read.csv('superstore.csv')

# Rename columns by replacing "." with "_"
colnames(df) <- gsub("\\.", "_", colnames(df))

head(df) 
```
#### Summarize data

```{r, echo=FALSE}
# Convert "Order.Date" to a date format
df$Order_Date <- mdy(df$Order_Date)

# Filter data for customers in 2014
customers_2014 <- df %>%
  filter(year(Order_Date) == 2014) %>%
  select(Customer_ID, Order_Date, Sales)

# Get IDs in 2014
distinct_customers_2014 <- distinct(customers_2014, Customer_ID)

# Select rows from 'superstore_data' where 'Customer_ID' is in 'distinct_customers_2014'
selected_rows <- df[df$Customer_ID %in% distinct_customers_2014$Customer_ID, ]


# Calculate Sales for each year
sales_2017 <- selected_rows %>%
  filter(year(Order_Date) == 2017) %>%
  group_by(Customer_ID) %>%
  summarize(Sales_2017 = sum(Sales))

sales_2016 <- selected_rows %>%
  filter(year(Order_Date) == 2016) %>%
  group_by(Customer_ID) %>%
  summarize(Sales_2016 = sum(Sales))

sales_2015 <- selected_rows %>%
  filter(year(Order_Date) == 2015) %>%
  group_by(Customer_ID) %>%
  summarize(Sales_2015 = sum(Sales))

sales_2014 <- selected_rows %>%
  filter(year(Order_Date) == 2014) %>%
  group_by(Customer_ID) %>%
  summarize(Sales_2014 = sum(Sales))

# Calculate Orders for each year
orders_2017 <- selected_rows %>%
  filter(year(Order_Date) == 2017) %>%
  group_by(Customer_ID) %>%
  summarize(Orders_2017 = n_distinct(Order_Date))

orders_2016 <- selected_rows %>%
  filter(year(Order_Date) == 2016) %>%
  group_by(Customer_ID) %>%
  summarize(Orders_2016 = n_distinct(Order_Date))

orders_2015 <- selected_rows %>%
  filter(year(Order_Date) == 2015) %>%
  group_by(Customer_ID) %>%
  summarize(Orders_2015 = n_distinct(Order_Date))

orders_2014 <- selected_rows %>%
  filter(year(Order_Date) == 2014) %>%
  group_by(Customer_ID) %>%
  summarize(Orders_2014 = n_distinct(Order_Date))

# Calculate Days since last order for each year
days_since_last_order_1117 <- selected_rows %>%
  filter(year(Order_Date) < 2017) %>%
  group_by(Customer_ID) %>%
  summarize(Days_since_last_order_1117 = as.numeric(difftime("2017-01-01", max(Order_Date), units = "days")))

days_since_last_order_1116 <- selected_rows %>%
  filter(year(Order_Date) < 2016) %>%
  group_by(Customer_ID) %>%
  summarize(Days_since_last_order_1116 = as.numeric(difftime("2016-01-01", max(Order_Date), units = "days")))

days_since_last_order_1115 <- selected_rows %>%
  filter(year(Order_Date) < 2015) %>%
  group_by(Customer_ID) %>%
  summarize(Days_since_last_order_1115 = as.numeric(difftime("2015-01-01", max(Order_Date), units = "days")))

# Merge all the calculated variables
summary_data <- selected_rows %>%
  select(Customer_ID) %>%
  left_join(sales_2017, by = "Customer_ID") %>%
  left_join(sales_2016, by = "Customer_ID") %>%
  left_join(sales_2015, by = "Customer_ID") %>%
  left_join(sales_2014, by = "Customer_ID") %>%
  left_join(orders_2017, by = "Customer_ID") %>%
  left_join(orders_2016, by = "Customer_ID") %>%
  left_join(orders_2015, by = "Customer_ID") %>%
  left_join(orders_2014, by = "Customer_ID") %>%
  left_join(days_since_last_order_1117, by = "Customer_ID") %>%
  left_join(days_since_last_order_1116, by = "Customer_ID") %>%
  left_join(days_since_last_order_1115, by = "Customer_ID") %>%
  distinct()

# Fill missing values with 0
summary_data[is.na(summary_data)] <- 0

# Print the summarized data
head(summary_data)
```


```{r, echo=FALSE}
dim(summary_data)

## Descriptive Analysis (Part II)

In this section, a descriptive analysis is conducted. Descriptive analysis involves examining the data to summarize its main characteristics, uncover patterns, and identify relationships between variables. This project performed univariate, bivariate, and multivariate descriptive analysis on the resulting summary_data dataframe.

### Univariate Analysis

Univariate analysis involves examining individual variables one at a time. In this assignment we provided the summary statistics, provided visualizations, and explored the distribution of each variable. The analysis is perfomed  on the Sales_2017, Sales_2016, Sales_2015, and Sales_2014 variables.

```{r, echo=FALSE}
# Univariate analysis for Sales_2017
summary(summary_data$Sales_2017)  # Summary statistics
```
```{r, echo=FALSE}
# Univariate analysis for Sales_2016
summary(summary_data$Sales_2016)  # Summary statistics
```

```{r, echo=FALSE}
# Univariate analysis for Sales_2016
summary(summary_data$Sales_2015)  # Summary statistics
```

```{r, echo=FALSE}
# Univariate analysis for Sales_2016
summary(summary_data$Sales_2014)  # Summary statistics
```


```{r, echo=FALSE}
# Create a 2x2 multi-panel plot
par(mfrow = c(2, 2))

# Univariate analysis for Sales_2017
summary_2017 <- summary_data$Sales_2017
hist(summary_2017, main = "Sales_2017 Distribution", xlab = "Sales_2017")
boxplot(summary_2017, main = "Sales_2017 Boxplot")

# Univariate analysis for Sales_2016
summary_2016 <- summary_data$Sales_2016
hist(summary_2016, main = "Sales_2016 Distribution", xlab = "Sales_2016")
boxplot(summary_2016, main = "Sales_2016 Boxplot")

# Univariate analysis for Sales_2015
summary_2015 <- summary_data$Sales_2015
hist(summary_2015, main = "Sales_2015 Distribution", xlab = "Sales_2015")
boxplot(summary_2015, main = "Sales_2015 Boxplot")

# Univariate analysis for Sales_2014
summary_2014 <- summary_data$Sales_2014
hist(summary_2014, main = "Sales_2014 Distribution", xlab = "Sales_2014")
boxplot(summary_2014, main = "Sales_2014 Boxplot")

# Reset the plotting parameters
par(mfrow = c(1, 1))  # Reset to a single panel

```


### Bivariate Analysis

Bivariate analysis examines the relationships between pairs of variables. In this assignment we calculate correlation coefficients, create scatterplots, and explore how two variables interact. Specifically we explore the bivariate analysis of Sales_2017 vs. Orders_2017, Sales_2016 vs. Orders_2016, Sales_2015 vs. Orders_2015, and Sales_2014 vs. Orders_2014

```{r, echo=FALSE}
# Create a 2x2 multi-panel plot
par(mfrow = c(2, 2))
par(mar = c(5, 5, 3, 2))  # Adjust the plot margins

# Bivariate analysis for Sales_2017 vs. Orders_2017
cor_2017 <- cor(summary_data$Sales_2017, summary_data$Orders_2017)  # Correlation coefficient
title_2017 <- paste("(Correlation =", round(cor_2017, 2), ")")
plot(summary_data$Sales_2017, summary_data$Orders_2017, 
     main = title_2017,
     xlab = "Sales_2017", ylab = "Orders_2017")

# Bivariate analysis for Sales_2016 vs. Orders_2016
cor_2016 <- cor(summary_data$Sales_2016, summary_data$Orders_2016)  # Correlation coefficient
title_2016 <- paste("(Correlation =", round(cor_2016, 2), ")")
plot(summary_data$Sales_2016, summary_data$Orders_2016, 
     main = title_2016,
     xlab = "Sales_2016", ylab = "Orders_2016")

# Bivariate analysis for Sales_2015 vs. Orders_2015
cor_2015 <- cor(summary_data$Sales_2015, summary_data$Orders_2015)  # Correlation coefficient
title_2015 <- paste("(Correlation =", round(cor_2015, 2), ")")
plot(summary_data$Sales_2015, summary_data$Orders_2015, 
     main = title_2015,
     xlab = "Sales_2015", ylab = "Orders_2015")

# Bivariate analysis for Sales_2014 vs. Orders_2014
cor_2014 <- cor(summary_data$Sales_2014, summary_data$Orders_2014)  # Correlation coefficient
title_2014 <- paste("(Correlation =", round(cor_2014, 2), ")")
plot(summary_data$Sales_2014, summary_data$Orders_2014, 
     main = title_2014,
     xlab = "Sales_2014", ylab = "Orders_2014")

# Reset the plotting parameters
par(mfrow = c(1, 1))  # Reset to a single panel


```

### Multivariate Analysis

Multivariate analysis explores relationships among three or more variables simultaneously. There are several techniques like multiple regression, principal component analysis (PCA), or clustering that can be used to conduct the multivariate analysis. Here we use multiple regression to predict Sales_2017 based on Sales_2016 and Orders_2017
```{r, echo=FALSE}
# Multivariate analysis: Multiple Regression
model <- lm(Sales_2017 ~ Sales_2016 + Orders_2017, data = summary_data)
summary(model)  # Summary of the regression model

```
Sales_2016: The p-value for Sales_2016 is 0.3134, which is greater than 0.05. This indicates that Sales_2016 is not statistically significant in predicting Sales_2017.
Orders_2017: The p-value for Orders_2017 is very close to zero (p-value: < 2.2e-16), indicating that Orders_2017 is highly statistically significant in predicting Sales_2017.

```{r, echo=FALSE}
colnames(summary_data)
```

## Regression Analysis
```{r, echo=FALSE}
# Remove Customer_ID column and set Sales_2017 as the dependent variable (DV)
summary_data <- summary_data[, !names(summary_data) %in% "Customer_ID"]
y <- summary_data$Sales_2017
x <- setdiff(names(summary_data), "Sales_2017")

# Create a new data frame with both y and x
model_data <- data.frame(y = y, summary_data[, x])

# Create a training dataset and testing dataset (e.g., 80% training, 20% testing)
set.seed(123)  # For reproducibility
train_indices <- createDataPartition(y, p = 0.8, list = FALSE)
train_data <- model_data[train_indices, ]
test_data <- model_data[-train_indices, ]

# Initialize a list to store model results
model_results <- list()
```

```{r, echo=FALSE}
# 1. Linear Regression
lm_model <- lm(formula = paste("y ~.", sep = "", collapse = "+"), data = train_data)
lm_predictions_train <- predict(lm_model, newdata = train_data)
lm_predictions_test <- predict(lm_model, newdata = test_data)
lm_r2 <- summary(lm_model)$r.squared
lm_rmse_train <- sqrt(mean((train_data$y - lm_predictions_train)^2))
lm_rmse_test <- sqrt(mean((test_data$y - lm_predictions_test)^2))
model_results$LinearRegression <- list(
  Coefficients = coef(lm_model),
  R_squared = lm_r2,
  Training_RMSE = lm_rmse_train,
  Testing_RMSE = lm_rmse_test
)
```

```{r, echo=FALSE}
# 2. Polynomial Regression (Example: Quadratic)
poly_model <- lm(formula = paste("y ~ poly(Sales_2016, 2) + Orders_2017"), data = train_data)
poly_predictions_train <- predict(poly_model, newdata = train_data)
poly_predictions_test <- predict(poly_model, newdata = test_data)
poly_r2 <- summary(poly_model)$r.squared
poly_rmse_train <- sqrt(mean((train_data$y - poly_predictions_train)^2))
poly_rmse_test <- sqrt(mean((test_data$y - poly_predictions_test)^2))
model_results$PolynomialRegression <- list(
  Coefficients = coef(poly_model),
  R_squared = poly_r2,
  Training_RMSE = poly_rmse_train,
  Testing_RMSE = poly_rmse_test
)
```

```{r, echo=FALSE}
# 3. Ridge Regression (L2 Regularization)
ridge_model <- cv.glmnet(x = as.matrix(train_data[, x]), y = train_data$y, alpha = 0)
ridge_predictions_train <- predict(ridge_model, s = 0.01, newx = as.matrix(train_data[, x]))
ridge_predictions_test <- predict(ridge_model, s = 0.01, newx = as.matrix(test_data[, x]))
ridge_r2 <- cor(ridge_predictions_train, train_data$y)^2
ridge_rmse_train <- sqrt(mean((train_data$y - ridge_predictions_train)^2))
ridge_rmse_test <- sqrt(mean((test_data$y - ridge_predictions_test)^2))
model_results$RidgeRegression <- list(
  Coefficients = as.vector(coef(ridge_model, s = 0.01)),
  R_squared = ridge_r2,
  Training_RMSE = ridge_rmse_train,
  Testing_RMSE = ridge_rmse_test
)
```

```{r, echo=FALSE}
# 4. Lasso Regression (L1 Regularization)
lasso_model <- cv.glmnet(x = as.matrix(train_data[, x]), y = train_data$y, alpha = 1)
lasso_predictions_train <- predict(lasso_model, s = 0.01, newx = as.matrix(train_data[, x]))
lasso_predictions_test <- predict(lasso_model, s = 0.01, newx = as.matrix(test_data[, x]))
lasso_r2 <- cor(lasso_predictions_train, train_data$y)^2
lasso_rmse_train <- sqrt(mean((train_data$y - lasso_predictions_train)^2))
lasso_rmse_test <- sqrt(mean((test_data$y - lasso_predictions_test)^2))
model_results$LassoRegression <- list(
  Coefficients = as.vector(coef(lasso_model, s = 0.01)),
  R_squared = lasso_r2,
  Training_RMSE = lasso_rmse_train,
  Testing_RMSE = lasso_rmse_test
)
```

```{r, echo=FALSE}
# 5. Random Forest Regression
rf_model <- randomForest(x = train_data[, x], y = train_data$y)
rf_predictions_train <- predict(rf_model, newdata = train_data[, x])
rf_predictions_test <- predict(rf_model, newdata = test_data[, x])
rf_r2 <- cor(rf_predictions_train, train_data$y)^2
rf_rmse_train <- sqrt(mean((train_data$y - rf_predictions_train)^2))
rf_rmse_test <- sqrt(mean((test_data$y - rf_predictions_test)^2))
model_results$RandomForest <- list(
  Importance = rf_model$importance,
  R_squared = rf_r2,
  Training_RMSE = rf_rmse_train,
  Testing_RMSE = rf_rmse_test
)
```

```{r, echo=FALSE}
# Model selection based on R-squared (choose the model with the highest R-squared on the test set)
best_model <- names(model_results)[which.max(sapply(model_results, function(x) x$R_squared))]

# Format the output for readability
formatted_results <- sapply(model_results, function(model) {
  c(
    Model = model$Model,
    R_squared_Train = round(model$R_squared, 4),
    Training_RMSE = round(model$Training_RMSE, 4),
    Testing_RMSE = round(model$Testing_RMSE, 4)
  )
})
```

```{r, echo=FALSE}
# Display the formatted results
formatted_results
```

```{r}
# Model selection based on RMSE
rmse_values <- sapply(model_results, function(model) model$Testing_RMSE)

# Print model results
for (model_name in names(model_results)) {
  cat("Model:", model_name, "\n")
  cat("Coefficients:", model_results[[model_name]]$Coefficients, "\n")
  cat("R-squared (Training):", model_results[[model_name]]$R_squared, "\n")
  cat("RMSE (Training):", model_results[[model_name]]$Training_RMSE, "\n")
  cat("RMSE (Testing):", model_results[[model_name]]$Testing_RMSE, "\n")
  cat("\n")
}
```
```{r, echo=FALSE}
# Select the model with the lowest RMSE
# Selectthe model with the highest R^2
cat("\nBest Model Selected:", best_model, "\n")
```
