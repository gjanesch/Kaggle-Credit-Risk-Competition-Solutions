# Solution attempt 1: Minimal logistic regression model
# Final submission: 2018-05-31
# Submission score: 0.690

# This is an attempt to get the simplest possible logistic regression model working - it uses only
# the application_train and application_test files, and the cleaning consists of just ditching
# columns with a lot of NAs, removing training rows with NAs, and filling in remaining NAs in the
# test data with that column's median data.

has_NAs <- function(x){
    return(sum(is.na(x)) > 0)
}

turn_into_onehot_column <- function(x, new_levels){
    return(new_levels[as.factor(x)])
}

train <- read.csv("./Data Files/application_train.csv")
test <- read.csv("./Data Files/application_test.csv")

# Lot of NAs in this data; for the first pass, we'll just ignore them
building_info_columns <- grep("_AVG$|_MODE$|_MEDI$", names(train))
other_cols_to_remove <- c("EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE")

train <- train[,-building_info_columns]
test <- test[,-(building_info_columns-1)] #subtract 1 to account for presence of TARGET column

train <- train[,!(names(train) %in% other_cols_to_remove)]
test <- test[,!(names(test) %in% other_cols_to_remove)]

# For this first try, just train on rows with full data on remaining columns
train <- train[!apply(train, FUN = has_NAs, MARGIN = 1),]

# There are still some missing values in test data; we'll just impute the median, though, since they
# are either only a couple in each column or 0/1 with one value predominant
test_cols_with_NAs <- names(test)[sapply(test, has_NAs)]
for(tc in test_cols_with_NAs){
    test[is.na(test[,tc]),tc] <- median(test[,tc], na.rm = TRUE)
}

model <- glm(TARGET ~ ., family = binomial(link = "logit"), data = train)
predictions <- predict(model, test, type = "response")
write.csv(data.frame(SK_ID_CURR = test$SK_ID_CURR, TARGET = round(predictions,3)),
          "v01_predictions.csv", row.names = FALSE)