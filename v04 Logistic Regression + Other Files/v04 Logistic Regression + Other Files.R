# Solution attempt 4: Logistic Regression + Other Files
# Final submission: 2018-06-14
# Submission score: 0.712

# In addition to new tweaks to the data, information from other files will be added in to try to
# improve predictions.

library(ROCR)
library(feather)
library(plyr)
library(dplyr)

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
TESTING <- TRUE

train <- readRDS("./Data Files/application_train.rds")
test <- readRDS("./Data Files/application_test.rds")

#################################################
# DATA CLEANING
#################################################

train_cleaning <- function(df){
    building_info_columns <- names(df)[grep("_AVG$|_MODE$|_MEDI$", names(df))]
    other_cols_to_remove <- c("EXT_SOURCE_1", "EXT_SOURCE_3", "OWN_CAR_AGE") #more NAs
    other_cols_to_remove <- c(other_cols_to_remove, "NAME_INCOME_TYPE", "NAME_HOUSING_TYPE")#large p-vals
    df <- df[,!(names(df) %in% c(building_info_columns,other_cols_to_remove))]
    
    return(df)
}

test_ID <- test$SK_ID_CURR

train <- train_cleaning(train)
test <- train_cleaning(test)

# For this first try, just train on rows with full data on remaining columns
train <- train[!apply(train, FUN = has_NAs, MARGIN = 1),]

#################################################
# DATA TRANSFORMATION
#################################################

train_transform <- function(df){
    #Make some things normal
    df$AMT_INCOME_TOTAL <- log10(df$AMT_INCOME_TOTAL)
    #df$AMT_CREDIT <- log10(df$AMT_CREDIT)
    df$CNT_CHILDREN <- as.factor(ifelse(df$CNT_CHILDREN <= 1 , as.character(df$CNT_CHILDREN), "2+"))
    return(df)
}

train <- train_transform(train)
test <- train_transform(test)

train[train$CODE_GENDER == "XNA","CODE_GENDER"] <- "F"
train$CODE_GENDER <- droplevels(train$CODE_GENDER)

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

train <- add_bureau_data(train, bureau_sub)
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
train <- add_prev_app_data(train, prev_app_sub)
test <- add_prev_app_data(test, prev_app_sub)
rm(previous_application, prev_app_sub)


cc_balance <- read_feather("./Data Files/credit_card_balance.feather")
cc_balance_sub <- cc_balance %>% group_by(SK_ID_CURR) %>%
    summarize(NUM_LATE_CC_PAYMENTS = sum(SK_DPD_DEF != 0), MAX_CREDIT_LIMIT = max(AMT_CREDIT_LIMIT_ACTUAL),
              NUM_PREV_CC_LOANS = length(unique(SK_ID_PREV)))

add_cc_balance_data <- function(df, cc_df){
    df <- left_join(df, cc_df, by = "SK_ID_CURR")
    no_cc_balance_data <- is.na(df$NUM_LATE_CC_PAYMENTS)
    for(clm in c("NUM_LATE_CC_PAYMENTS","MAX_CREDIT_LIMIT", "NUM_PREV_CC_LOANS")){
        df[no_cc_balance_data, clm] <- 0
    }
    return(df)
}
train <- add_cc_balance_data(train, cc_balance_sub)
test <- add_cc_balance_data(test, cc_balance_sub)
rm(cc_balance, cc_balance_sub)

install_paym <- read_feather("./Data Files/installments_payments.feather")
missing_payment_info <- is.na(install_paym$AMT_PAYMENT)
install_paym[missing_payment_info,"AMT_PAYMENT"] <- install_paym[missing_payment_info,"AMT_INSTALMENT"]
install_paym[missing_payment_info,"DAYS_ENTRY_PAYMENT"] <- install_paym[missing_payment_info,"DAYS_INSTALMENT"]
install_paym_sub <- install_paym %>% group_by(SK_ID_CURR) %>%
    summarize(NUM_PAYMENTS_UNDER = sum(AMT_PAYMENT < AMT_INSTALMENT),
              NUM_PAYMENTS_LATE = sum(DAYS_ENTRY_PAYMENT > DAYS_INSTALMENT))

add_install_paym <- function(df, ip_df){
    df <- left_join(df, ip_df, by = "SK_ID_CURR")
    df[is.na(df$NUM_PAYMENTS_UNDER), "NUM_PAYMENTS_UNDER"] <- 0
    df[is.na(df$NUM_PAYMENTS_LATE), "NUM_PAYMENTS_LATE"] <- 0
    return(df)
}

train <- add_install_paym(train, install_paym_sub)
test <- add_install_paym(test, install_paym_sub)

rm(install_paym, install_paym_sub, missing_payment_info)

POS_cash <- read_feather("./Data Files/POS_CASH_balance.feather")
pc_sub <- POS_cash %>% group_by(SK_ID_CURR) %>% summarize(NUM_LATE_POS_PAYMENTS = sum(SK_DPD_DEF == 1))

add_POS <- function(df, pos_df){
    df <- left_join(df, pos_df, by = "SK_ID_CURR")
    df[is.na(df$NUM_LATE_POS_PAYMENTS), "NUM_LATE_POS_PAYMENTS"] <- 0
    return(df)
}
train <- add_POS(train, pc_sub)
test <- add_POS(test, pc_sub)
rm(POS_cash, pc_sub)


#################################################
# MAKING THE MODEL
#################################################

train$SK_ID_CURR <- NULL
test$SK_ID_CURR <- NULL

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
              "v04_predictions.csv", row.names = FALSE)
}
print_summary_to_file(model, "v04_model_summary.txt")

