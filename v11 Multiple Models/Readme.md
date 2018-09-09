This was the 11th and final attempt at coming up with a solution.  This version was a quick attempt to employ model averaging, as LightGBM seemed to be about as strong as I could make it with limited time left in the competition.

The solution was partly a return to previous versions, as in addition to LightGBM it brought back XGBoost (in Python this time) and Keras as parts of the solution, along with an attempt to include AdaBoost from <tt>scikit-learn</tt>.  It also attempted to try using some form of Bayesian optimization to find the best way to combine the solutions, though since LightGBM performed the best of the group, it was always heavily weighted against the other solutions.

Despite the short timeframe for experimenting with averaging, there was a slight improvement in score.

Best public leaderboard AUC: 0.79522
