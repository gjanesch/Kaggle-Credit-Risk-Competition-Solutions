This is the ninth attempt at a solution.  Unlike the previous solution, this one was not just an iterative improvement on a basic LightGBM model, as this included multiple changes.

Two changes were fairly minor.  First, another attempt at Bayesian optimization of hyperparameters, this time using <tt>scikit-optimize</tt>.  The other is that instead of iterating through each supplementary file and adding results onto the full data set one at a time, the creation of each supplementary file's dataframe was enclosed inside a function, and then all the resulting dataframes were joined.  This was to make the code a bit more modular, so that the full sequence would not need to be rerun every time a change was made to one of the files' feature engineering.

The more major change was the inclusion of a logistic regression model built in Tensorflow.  Regressions were run on the main train data as well as the individual supplementary dataframes, and then were added to the LightGBM models as features.  Feature importance results indicated that these were fairly commonly used features, with importance scores of several hundred (with the largest scores running between about 1500-2200).

Despite all of this, improvement over v08 was marginal.

Best public leaderboard AUC: 0.78925
