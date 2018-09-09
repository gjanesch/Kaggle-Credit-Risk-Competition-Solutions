# Solution attempt #06: XGBoost starting from v04 data cleaning
# Final submission: 2018-06-22
# Final submission score: 0.727

# First crack at xgboost.  Started from v04 data cleaning, but incorporated more stuff over time.

library(ROCR)
library(feather)
library(plyr)
library(dplyr)
library(caret)
library(xgboost)

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

train_data <- readRDS("./Data Files/application_train.rds")
test <- readRDS("./Data Files/application_test.rds")

#################################################
# DATA CLEANING
#################################################

train_cleaning <- function(df){
    building_info_columns <- names(df)[grep("_AVG$|_MODE$|_MEDI$", names(df))]
    other_cols_to_remove <- c("EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE") #more NAs
    other_cols_to_remove <- c(other_cols_to_remove, "FLAG_DOCUMENT_2")#low importance
    df <- df[,!(names(df) %in% c(building_info_columns,other_cols_to_remove))]
    
    return(df)
}

test_ID <- test$SK_ID_CURR

train_data <- train_cleaning(train_data)
test <- train_cleaning(test)

# For this first try, just train on rows with full data on remaining columns
train_data <- train_data[!apply(train_data, FUN = has_NAs, MARGIN = 1),]

#################################################
# FEATURE ENGINEERING
#################################################

feature_engineering <- function(df){
    #df$CREDIT_INCOME_RATIO <- log10(df$AMT_CREDIT/df$AMT_INCOME_TOTAL)
    #df$ANNUITY_INCOME_RATIO <- log10(df$AMT_ANNUITY/df$AMT_INCOME_TOTAL)
    #df$INCOME_PER_HEAD <- log10(df$AMT_INCOME_TOTAL/df$CNT_FAM_MEMBERS)
    df$CREDIT_INCOME_RATIO <- df$AMT_CREDIT/df$AMT_INCOME_TOTAL
    df$ANNUITY_INCOME_RATIO <- df$AMT_ANNUITY/df$AMT_INCOME_TOTAL
    df$INCOME_PER_HEAD <- df$AMT_INCOME_TOTAL/df$CNT_FAM_MEMBERS
    df$IS_EMPLOYED <- 1 * (df$DAYS_EMPLOYED < 0)
    return(df)
}

train_data <- feature_engineering(train_data)
test <- feature_engineering(test)

#################################################
# DATA TRANSFORMATION
#################################################

train_transform <- function(df){
    #Make some things normal
    #df$AMT_INCOME_TOTAL <- log10(df$AMT_INCOME_TOTAL)
    #df$AMT_CREDIT <- log10(df$AMT_CREDIT)
    
    df$CNT_CHILDREN <- as.factor(ifelse(df$CNT_CHILDREN <= 1 , as.character(df$CNT_CHILDREN), "2+"))
    df$FLAG_OWN_CAR <- 1 * (df$FLAG_OWN_CAR == "Y")
    df$FLAG_OWN_REALTY <- 1 * (df$FLAG_OWN_REALTY == "Y")
    
    #There are several variables which don't contain actual NAs, but have values that seem to act
    #as the equivalent
    df[df$DAYS_EMPLOYED > 0, "DAYS_EMPLOYED"] <- 0
    
    return(df)
}

train_data <- train_transform(train_data)
test <- train_transform(test)

train_data[train_data$CODE_GENDER == "XNA","CODE_GENDER"] <- "F"
train_data$CODE_GENDER <- droplevels(train_data$CODE_GENDER)

# There are still some missing values in test data; we'll just impute the median, though, since they
# are either only a couple in each column or 0/1 with one value predominant
test_cols_with_NAs <- names(test)[sapply(test, has_NAs)]
for(tc in test_cols_with_NAs){
    test[is.na(test[,tc]),tc] <- median(test[,tc], na.rm = TRUE)
}


#################################################
# ADDING NEW FILES
#################################################

bureau <- read_feather("./Data Files/bureau.feather")
bureau_sub <- bureau %>% group_by(SK_ID_CURR) %>%
    summarize(CREDIT_COUNT = n(), ANY_OVERDUE = (max(CREDIT_DAY_OVERDUE == 0)))

add_bureau_data <- function(df, bureau_df){
    df <- left_join(df, bureau_df, by = "SK_ID_CURR")
    df[is.na(df$CREDIT_COUNT), "CREDIT_COUNT"] <- 0
    df[is.na(df$ANY_OVERDUE), "ANY_OVERDUE"] <- 0
    return(df)
}

train_data <- add_bureau_data(train_data, bureau_sub)
test <- add_bureau_data(test, bureau_sub)

rm(bureau, bureau_sub)

previous_application <- read_feather("./Data Files/previous_application.feather")
prev_app_sub <- previous_application %>% group_by(SK_ID_CURR) %>%
    summarize(NUMBER_APPLICATIONS = n(), NUMBER_REFUSED = sum(NAME_CONTRACT_STATUS == "Refused"),
              NUMBER_APPROVED = sum(NAME_CONTRACT_STATUS == "Approved"))

add_prev_app_data <- function(df, prev_app_df){
    df <- left_join(df, prev_app_df, by = "SK_ID_CURR")
    no_previous_applications <- is.na(df$NUMBER_APPLICATIONS)
    for(clm in c("NUMBER_APPLICATIONS", "NUMBER_REFUSED","NUMBER_APPROVED")){
        df[no_previous_applications,clm] <- 0
    }
    return(df)
}
train_data <- add_prev_app_data(train_data, prev_app_sub)
test <- add_prev_app_data(test, prev_app_sub)
rm(previous_application, prev_app_sub)


cc_balance <- read_feather("./Data Files/credit_card_balance.feather")
cc_balance_sub <- cc_balance %>% group_by(SK_ID_CURR) %>%
    summarize(NUM_LATE_CC_PAYMENTS = sum(SK_DPD_DEF != 0), MAX_CREDIT_LIMIT = max(AMT_CREDIT_LIMIT_ACTUAL),
              NUM_PREV_CC_LOANS = length(unique(SK_ID_PREV)), MAX_BALANCE = max(AMT_BALANCE), AVG_CC_BALANCE = mean(AMT_BALANCE))

add_cc_balance_data <- function(df, cc_df){
    df <- left_join(df, cc_df, by = "SK_ID_CURR")
    no_cc_balance_data <- is.na(df$NUM_LATE_CC_PAYMENTS)
    for(clm in c("NUM_LATE_CC_PAYMENTS","MAX_CREDIT_LIMIT", "NUM_PREV_CC_LOANS", "MAX_BALANCE", "AVG_CC_BALANCE")){
        df[no_cc_balance_data, clm] <- 0
    }
    return(df)
}
train_data <- add_cc_balance_data(train_data, cc_balance_sub)
test <- add_cc_balance_data(test, cc_balance_sub)
rm(cc_balance, cc_balance_sub)

install_paym <- read_feather("./Data Files/installments_payments.feather")
missing_payment_info <- is.na(install_paym$AMT_PAYMENT)
install_paym[missing_payment_info,"AMT_PAYMENT"] <- install_paym[missing_payment_info,"AMT_INSTALMENT"]
install_paym[missing_payment_info,"DAYS_ENTRY_PAYMENT"] <- install_paym[missing_payment_info,"DAYS_INSTALMENT"]
install_paym_sub <- install_paym %>% group_by(SK_ID_CURR) %>%
    summarize(NUM_PAYMENTS_UNDER = sum(AMT_PAYMENT < AMT_INSTALMENT),
              NUM_PAYMENTS_LATE = sum(DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT),
              MIN_PAYMENT = min(AMT_PAYMENT), MAX_PAYMENT = max(AMT_PAYMENT),
              BEST_PAYMENT_DATE = max(DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT),
              WORST_PAYMENT_DATE = min(DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT))

add_install_paym <- function(df, ip_df){
    df <- left_join(df, ip_df, by = "SK_ID_CURR")
    df[is.na(df$NUM_PAYMENTS_UNDER), "NUM_PAYMENTS_UNDER"] <- 0
    df[is.na(df$NUM_PAYMENTS_LATE), "NUM_PAYMENTS_LATE"] <- 0
    df[is.na(df$MAX_PAYMENT), "MAX_PAYMENT"] <- 0
    df[is.na(df$MIN_PAYMENT), "MIN_PAYMENT"] <- 0
    df[is.na(df$BEST_PAYMENT_DATE), "BEST_PAYMENT_DATE"] <- 0
    df[is.na(df$WORST_PAYMENT_DATE), "WORST_PAYMENT_DATE"] <- 0
    return(df)
}

train_data <- add_install_paym(train_data, install_paym_sub)
test <- add_install_paym(test, install_paym_sub)

rm(install_paym, install_paym_sub, missing_payment_info)

POS_cash <- read_feather("./Data Files/POS_CASH_balance.feather")
pc_sub <- POS_cash %>% group_by(SK_ID_CURR) %>% summarize(NUM_LATE_POS_PAYMENTS = sum(SK_DPD_DEF == 1))

add_POS <- function(df, pos_df){
    df <- left_join(df, pos_df, by = "SK_ID_CURR")
    df[is.na(df$NUM_LATE_POS_PAYMENTS), "NUM_LATE_POS_PAYMENTS"] <- 0
    return(df)
}
train_data <- add_POS(train_data, pc_sub)
test <- add_POS(test, pc_sub)
rm(POS_cash, pc_sub)


#################################################
# MAKING THE MODEL
#################################################

train_data$SK_ID_CURR <- NULL
test$SK_ID_CURR <- NULL


xgb_training <- function(training_df){
    cv.ctrl <- trainControl(method = 'repeatedcv', repeats = '1', number = 3, verboseIter = TRUE, classProbs = TRUE)
    xgb.grid <- expand.grid(nrounds = c(450,500,1000),  #450
                            eta = c(0.05),
                            max_depth = c(4), #4
                            gamma = c(0.48), # 0.48
                            colsample_bytree = c(0.75),   #0.8
                            min_child_weight = 1,
                            subsample = c(0.75))
    model <- train(TARGET ~ ., data = training_df, method = 'xgbTree', trControl = cv.ctrl, tuneGrid = xgb.grid, verbose = T, objective = "binary:logistic")
    return(model)
}

if(TESTING == TRUE){
    set.seed(555)
    train_indices <- sample(seq_len(nrow(train_data)), size = floor(0.7*nrow(train_data)))
    train_train <- train_data[train_indices,]
    train_test <- train_data[-train_indices,]
    model <- xgb_training(train_train)
    predictions <- predict(model, train_test)
    #predictions <- predict(model, train_test, type = "response")
    prediction_object <- prediction(predictions, train_test$TARGET)
    
    auc <- performance(prediction_object, measure = "auc")
    auc <- auc <- auc@y.values[[1]]
    print(paste("AUC: ",auc))
}else{
    model <- xgb_training(train_data)
    predictions <- predict(model, test)
    write.csv(data.frame(SK_ID_CURR = test_ID, TARGET = round(predictions,3)),
              "v06_predictions_after.csv", row.names = FALSE)
}
print_summary_to_file(model, "v06_model_summary.txt")

