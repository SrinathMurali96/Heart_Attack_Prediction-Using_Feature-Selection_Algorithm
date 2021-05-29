# Heart_Attack_Prediction-Using_Feature-Selection_Algorithm

**Feature selection** is the process of reducing the number of input variables when developing a predictive model.

Top reasons to use feature selection are:
    It enables the machine learning algorithm to train faster.
    It reduces the complexity of a model and makes it easier to interpret.
    It improves the accuracy of a model if the right subset is chosen.
    It reduces overfitting.
 
 The two feature selection methods used are:
        **1. Recursive Feature Elimination 
          2. SelectKBest**
          
 **Recursive Feature Elimination :**

The goal of recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and the importance of each feature is obtained either through a coef_ attribute or through a feature_importances_ attribute.

Then, the least important features are pruned from the current set of features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.

**For refernce : http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html **

**SelectKBest**
  Select features according to the k highest scores.
  
  f_classif
  ANOVA F-value between label/feature for classification tasks.

  f_regression
  F-value between label/feature for regression tasks.

**For refernce : http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html **
