This was the sixth attempt at a solution to the Credit Risk competition, this one using XGBoost in R.  The feature engineering started from the final state of the logistic regression in v04.  It was developed in parallel with v05, and took most of the attention due to familiarity and the handling of NAs.

This version involved much more feature engineering than previous attempts, although due to the <tt>summary()</tt> function not behaving the same for XGBoost as it did for logistic regression, breakdowns of the model weren't used.

Best public leaderboard AUC: 0.72924
