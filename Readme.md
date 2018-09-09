This is a sequence of solutions developed in working on <a href="">Kaggle's Home Credit Default Risk</a> competition.  The sequence was developed over the course of about three months (June to August 2018).

It began with a few easy-to-set-up models - logistic regression and neural networks in Keras - but eventually turned to using gradient-boosted trees, primarily through LightGBM.  The solutions are mostly written in Jupyter notebooks in Python, though the logistic regression and XGBoost solutions were in R.

Final score: 0.78900
Final ranking: 3308/7198 (54th percentile)


Solution score history:
| Version | Model Type           | Best Public AUC |
| ------- | -------------------- | --------------- |
| v01     | Logistic Regression  | 0.69096         |
| v02     | Logistic Regression  | 0.69020         |
| v03     | Neural Network       | 0.68599         |
| v04     | Logistic Regression  | 0.71264         |
| v05     | Neural Network       | 0.72516         |
| v06     | XGBoost              | 0.72924         |
| v07     | LightGBM             | 0.76718         |
| v08     | LightGBM             | 0.78849         |
| v09     | LightGBM + Log. Reg. | 0.78925         |
| v10     | LightGBM + Log. Reg. | 0.79279         |
| v11     | Multiple models      | 0.79522         |
