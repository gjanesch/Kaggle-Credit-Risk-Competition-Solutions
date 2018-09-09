This is the seventh solution attempt for the competition.  It is the first one to use LightGBM, and is essentially a translation of the final xgboost solution, with a limited amount of additional feature engineering plus code the outputs the feature importances from training.

This version also attempted to employ some Bayesian optimization of the hyperparameters using the <tt>BayesianOptimization</tt> library.  Due to speed issues, the optimization code was not used beyond confirming that it worked.

Though the model showed some immediate improvement over the XGBoost attempt, the biggest improvement came from the inclusion of the columns EXT_SOURCE_1, _2, and _3, which had been ignored in previous solutions due to the higher-than-desired proportion of missing values.

Best public leaderboard AUC: 0.76718
