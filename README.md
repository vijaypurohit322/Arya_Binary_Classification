## Objective: Binary-Classification
*Assignment from [Arya.ai](https://arya.ai/)*
- - - - - - - -
## Files
- [EDA](https://github.com/gshashank84/Binary-Classification/blob/master/EDA.ipynb)
- [Predictions of Test Data](https://github.com/gshashank84/Binary-Classification/blob/master/Predictions.ipynb)
- [Print Performance of Models](https://github.com/gshashank84/Binary-Classification/blob/master/performance_print.py)
- [Requirements.txt](https://github.com/gshashank84/Binary-Classification/blob/master/requirements.txt)
- - - - - - -

**Train Dataset Shape -> (3909,58)**   
**Train Dataset Shape -> (690,58)**

*Dataset: Sparse and High Dimensional*
- - - - - - - - 
## Key Steps:
1. Used RandomForest Classifier for feature selection.
2. Selected top 30 features with respect to their feature importance.
3. For metric I have considered Binary CrossEntropy and AUC score.
4. The best model I get is Xgboost.
