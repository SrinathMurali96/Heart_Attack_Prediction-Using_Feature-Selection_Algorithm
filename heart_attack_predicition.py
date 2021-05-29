import logging
import xgboost 
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression, f_classif
from pandas.api.types import is_string_dtype
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

LOGGER = logging.getLogger(__name__)

class Model:
    
    def feature_selection(self,df,target,feature_type):
        LOGGER.info('Feature Automation Method - Starts')
        try:
                df = df.fillna(0)
                le = LabelEncoder()
                y = df[target]
                label_encoder = {}
                df.drop(target, axis=1, inplace=True)
                str_type = is_string_dtype(y)
                y = le.fit_transform(y)
                
                x_axis = []
                y_axis = []
                
                for i in df.columns:
                    # To deselect column having unique values
                    unique_col_length = len(pd.unique(df[i]))
                    total_length = len(df[i])
                    percentage = int((unique_col_length/total_length)*100)
                    if df[i].is_unique or len(pd.unique(df[i])) == 1 or percentage > 80: 
                        df.drop(i, axis=1, inplace=True)
                 
                # Applying Label Encoding to categorical columns   
                for col in df.columns.values:
                    if df[col].dtype not in ['float64', 'int32']:
                        label_encoder[col] = LabelEncoder()
                        df[col] = df[col].astype(str)
                        label_encoder[col] = label_encoder[col].fit(df[col])
                        df[col] = label_encoder[col].transform(df[col].astype(str))
                
                # Feature selection with Select K Best 
                if feature_type == "SelectKBest":
                    if str_type:
                        # For classification type
                        lr = f_classif       
                    else:
                        # For Regression type
                        lr = f_regression  
                    
                    kbest = SelectKBest(score_func=lr, k=1).fit(df, y)
                    rank_dic = {}
                    for i, k in enumerate(df.columns.values):
                        rank_dic[k] = kbest.scores_[i]
                        
                # Feature selection with RFE
                else:
                    if str_type:
                        # For classification type
                        lr = DecisionTreeClassifier()
                    else:
                        # For Regression type
                        lr = DecisionTreeRegressor()
                    rfe = RFE(lr, n_features_to_select=1)
                    rfe.fit(df, y)
                    rank_dict = {}
                    
                    for i, k in enumerate(df.columns.values):
                        rank_dict[k] = rfe.ranking_[i]
    
                sorted_dict = sorted(rank_dict.items(), key=lambda x: x[1])                
                rank_dict = dict(sorted(rank_dict.items(), key=lambda x: x[1]))

                significant_list = []
                if feature_type == 'RFE':  
                    length = int(len(rank_dict) * 0.6)
                    for i in range(0, length):
                        significant_list.append(dict_list[i])
                

                elif feature_type == 'SelectKBest':
                    length = int(len(rank_dict) * 0.6)
                    for i in range(0, length):
                        significant_list.append(dict_list[i])
                        
                LOGGER.info('Feature Automation Method - Ends')
                return significant_list
            
    
        except Exception:
            LOGGER.error(traceback.format_exc())
            return 'Failure'
        
        
        
    def model_training(self):
        LOGGER.info('Model Training  Method - Starts')
        try:
            dataframe = pd.read_csv(r'C:\Users\Srinath.nm\Downloads\archive\heart.csv')
            
            # Finding the Correlation between data using Heatmap chart
            plt.figure(figsize=(15,10))
            sns.heatmap(dataframe.corr(), annot=True)
            
            target = 'Heart Attack Output'
            feature_type = 'RFE'
            significant_attributes = self.feature_selection(dataframe,target,feature_type)
            
            X = dataframe[significant_attributes]
            y = dataframe['Heart Attack Output']
            
            # Applying Normalisation and encoding techniques using StandardScaler and LabelEncoder
            label_encoder = {}
            for col in X.columns.values:
                if X[col].dtype not in ['float64', 'int32']:
                    label_encoder[col] = LabelEncoder()
                    X[col] = X[col].astype(str)
                    label_encoder[col] = label_encoder[col].fit(X[col])
                    X[col] = label_encoder[col].transform(X[col].astype(str))
            
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)
            
            # Model Training starts 
            key = ['LogisticRegression','KNeighborsClassifier','SVC','RandomForestClassifier','XGBClassifier','AdaBoostClassifier']
            value = [LogisticRegression(),KNeighborsClassifier(),SVC(),RandomForestClassifier(),xgboost.XGBClassifier(),AdaBoostClassifier()]
            model = dict(zip(key,value))
            
            predicted =[]
            max_accuracy = 0
            best_algorithm = ''
            for model_name,classifier in model.items():
                model=classifier
                model.fit(X_train,y_train)
                prediction = model.predict(X_test)
                score = accuracy_score(y_test, prediction)
                predicted.append(score)
                print(model_name.upper(),':',score*100)
                if score>max_accuracy:
                    best_algorithm = model_name.upper()
                    max_accuracy = score
            # Visualizing the accuracy's in bar chart
            plt.figure(figsize = (15,8))
            sns.barplot(x = key, y = predicted)
        
            LOGGER.info('Model Training Method - Ends')
            return best_algorithm,max_accuracy
            
    
        except Exception:
            LOGGER.error(traceback.format_exc())
            return 'Failure'
            
        
obj = Model()
best_algorithm,max_accuracy = obj.model_training()
print("The {} algorithm has the maximum accuracy of {} %".format(best_algorithm,max_accuracy*100 ))
        
        
