library(feather)
library(plyr)
library(dplyr)
library(ROCR)

#cc base AUC: 0.6632

target_df <- read_feather("target.feather")
train_ids <- target_df$SK_ID_CURR
target <- target_df$TARGET

print_summary_to_file <- function(model, file){
    sink(file)
    print(summary(model))
    sink()
}

has_NAs <- function(x){
    return(sum(is.na(x)) > 0)
}

# inner join by SK_ID_CURR
prepare_df <- function(df, target_df){
    df <- inner_join(df, target_df, by = "SK_ID_CURR")
    df <- df[,!sapply(df, has_NAs)]
    return(df)
}

polynomial_test <- function(df, degree = 2){
    df_features <- df %>% select(-SK_ID_CURR)
    df_feat_poly <- df_features ^ degree
    #names(df_feat_poly) <- paste0(names(df_feat_poly),"_DEG",degree)
    return(cor(df_feat_poly)[,"TARGET"])
}

variance_inflation_factors <- function(df){
    
    vif.df <- data.frame(VarName = character(0), Rsquared = numeric(0))
    
    numeric.columns <- names(df)[sapply(df, is.numeric)]
    for(colname in numeric.columns){
        new_formula <- paste(colname, '~ .', sep = "")
        model <- lm(new_formula, data = df)
        R2 <- summary.lm(model)$r.squared
        vif.df <- rbind(vif.df, data.frame(VarName = colname, Rsquared = 1/(1-R2)))
    }
    return(vif.df)
}

log_regress_train <- function(df, num_models = 4, model_file = "v10_model_summary.txt"){
    
    df$SK_ID_CURR <- NULL
    auc_vec <- numeric(0)
    for(x in 1:num_models){
        train_indices <- sample(seq_len(nrow(df)), size = floor(0.7*nrow(df)))
        train_train <- df[train_indices,]
        train_test <- df[-train_indices,]
        model <- glm(TARGET ~ ., family = binomial(link = "logit"), data = train_train)
        predictions <- predict(model, train_test, type = "response")
        prediction_object <- prediction(predictions, train_test$TARGET)
        
        auc <- performance(prediction_object, measure = "auc")
        auc <- auc@y.values[[1]]
        print(paste("AUC:",auc))
        auc_vec <- c(auc_vec,auc)
    }
    print_summary_to_file(model, model_file)
    
    print(paste("Avg AUC: ",mean(auc_vec)))
}