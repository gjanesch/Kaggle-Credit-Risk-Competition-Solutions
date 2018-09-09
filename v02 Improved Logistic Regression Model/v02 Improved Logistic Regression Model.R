# Solution attempt 2: Improved logistic regression model
# Final submission: 2018-06-10
# Submission score: 0.689

# This logistic regression is intended to be tuned based on a more thorough examination of the data
# and use of metrics to check model performance (especially AUC, since that's how Kaggle scores
# the competition).

library(ROCR)

has_NAs <- function(x){
    return(sum(is.na(x)) > 0)
}

turn_into_onehot_column <- function(x, new_levels){
    return(new_levels[as.factor(x)])
}

print_summary_to_file <- function(model, file){
    sink(file)
    print(summary(model))
    sink()
}

# Should be used as a flag in order to control whether I'm just testing things or making a
# submission
TESTING <- FALSE

train <- readRDS("./Data Files/application_train.rds")
test <- readRDS("./Data Files/application_test.rds")

#################################################
# DATA CLEANING
#################################################

# Lot of NAs in this data; for the first pass, we'll just ignore them
building_info_columns <- grep("_AVG$|_MODE$|_MEDI$", names(train))
train <- train[,-building_info_columns]
test <- test[,-(building_info_columns-1)] #subtract 1 to account for presence of TARGET column
test_ID <- test$SK_ID_CURR

train$SK_ID_CURR <- NULL
test$SK_ID_CURR <- NULL

# Columns to remove based on large numbers of NAs
other_cols_to_remove <- c("EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE")

# Additional columns removed due to very high z-scores in original model
other_cols_to_remove <- c(other_cols_to_remove, "NAME_INCOME_TYPE", "NAME_HOUSING_TYPE")

train <- train[,!(names(train) %in% other_cols_to_remove)]
test <- test[,!(names(test) %in% other_cols_to_remove)]


train$NAME_INCOME_TYPE <- NULL
test$NAME_INCOME_TYPE <- NULL

# For this first try, just train on rows with full data on remaining columns
train <- train[!apply(train, FUN = has_NAs, MARGIN = 1),]

#################################################
# DATA TRANSFORMATION
#################################################

# Make some things normal
train$AMT_INCOME_TOTAL <- log10(train$AMT_INCOME_TOTAL)
test$AMT_INCOME_TOTAL <- log10(test$AMT_INCOME_TOTAL)
#train$AMT_CREDIT <- log10(train$AMT_CREDIT)
#test$AMT_CREDIT <- log10(test$AMT_CREDIT)

#train[train$CODE_GENDER == "XNA",] <- "F"

# CNT_CHILDREN as a numeric has p-values around 0.1; chi^2 test for below transformation vs TARGET
# produces chi-squared statistic of 140ish
train$CNT_CHILDREN <- as.factor(ifelse(train$CNT_CHILDREN <= 1 , as.character(train$CNT_CHILDREN), "2+"))
test$CNT_CHILDREN <- as.factor(ifelse(test$CNT_CHILDREN <= 1, as.character(test$CNT_CHILDREN), "2+"))

# There are still some missing values in test data; we'll just impute the median, though, since they
# are either only a couple in each column or 0/1 with one value predominant
test_cols_with_NAs <- names(test)[sapply(test, has_NAs)]
for(tc in test_cols_with_NAs){
    test[is.na(test[,tc]),tc] <- median(test[,tc], na.rm = TRUE)
}

if(TESTING == TRUE){
    set.seed(555)
    train_indices <- sample(seq_len(nrow(train)), size = floor(0.7*nrow(train)))
    train_train <- train[train_indices,]
    train_test <- train[-train_indices,]
    model <- glm(TARGET ~ ., family = binomial(link = "logit"), data = train_train)
    predictions <- predict(model, train_test, type = "response")
    prediction_object <- prediction(predictions, train_test$TARGET)
    
    auc <- performance(prediction_object, measure = "auc")
    auc <- auc <- auc@y.values[[1]]
    print(paste("AUC: ",auc))
}else{
    model <- glm(TARGET ~ ., family = binomial(link = "logit"), data = train)
    predictions <- predict(model, test, type = "response")
    write.csv(data.frame(SK_ID_CURR = test_ID, TARGET = round(predictions,3)),
              "v02_predictions.csv", row.names = FALSE)
}
print_summary_to_file(model, "v02_model_summary.txt")

# AUC 4+: 0.704479