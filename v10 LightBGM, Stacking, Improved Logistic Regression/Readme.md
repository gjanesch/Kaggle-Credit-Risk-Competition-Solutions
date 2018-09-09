This is the tenth solution attempt for the competition.  It follows up on the changes made in the previous version, and makes a couple additional changes.

The obvious change, from looking at the files in the directory, is that the supplementary data files' code were all split into individual notebooks, along with splitting common functions off into their own file so that they can be imported and keep the notebooks more concise.  This is essentially a continuation of the reorganizing of the code from the previous attempt.

It also attempts to try to improve the utility of the logistic regressions added for stacking.  The Tensorflow implementation was dropped in favor of the sklearn version, which seemed to be more reliable.  It also attempted to automate a search for polynomial transformations of variables that would correlate better with the predictions in order to improve the scores, as well as picking out variables apparently unimportant to the logistic regression.

Improvements were modest.

Best public leaderboard AUC: 0.79279
