#libraries 

import yaml
import argparse

import wandb

import catboost 
import xgboost 
import lightgbm

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, f1_score 
from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import time
import numpy as np 
import re 


def main():
    
    #--------------Set up your dataset path--------------#
    #support cmd line arguments using argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Value for dataset")
    parser.add_argument("--algorithm", help="Value for algorithm")
    parser.add_argument("--sweep_method", help="Value for sweep_method")    
    args = parser.parse_args()
    
    #assign arguments to local variables 
    algorithm_name = args.algorithm
    dataset_name = args.dataset
    method_name = args.sweep_method

    if algorithm_name == "XGBClassifier":
                params = {
                            "n_estimators": {
                                "values": [100, 600, 1100, 1600, 2100, 2600, 3100, 3600, 4000]
                            },
                            "eta": {
                                "values": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
                            },
                            "max_depth": {
                                "values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                            },
                            "subsample": {
                                "values": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                            },  
                            "colsample_bytree": {
                                "values": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                            },
                            "colsample_bylevel": {
                                "values": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
                            },
                            "min_child_weight": {
                                "values": [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e1]
                            },
                            "alpha": {
                                "values": [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e1]
                            }, 
                            "lambda": {
                                "values": [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e1]
                            },
                            "gamma": {
                                "values": [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e1]
                            }
                         }

    elif algorithm_name == "LGBMClassifier":
                params = {
                            "class_weight": {
                                "values": [None, 'balanced']
                            },
                            "boosting_type": {
                                "values": ['gbdt', 'dart', 'goss']
                            },
                            "num_leaves": {
                                "values": [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
                            },
                            "learning_rate": {
                                "values": [0.01, 0.03, 0.05, 0.07, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.2]
                            },  
                            "subsample_for_bin": {
                                "values": [20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000, 180000, 200000, 220000, 240000, 260000, 280000, 300000]
                            },
                            "feature_fraction": {
                                "values": [0.5, 0.6, 0.7, 0.8, 0.9, 1]
                            },
                            "bagging_fraction": {
                                "values": [0.5, 0.6, 0.7, 0.8, 0.9, 1]
                            },
                            "min_data_in_leaf": {
                                "values": [0, 1, 2, 3, 4, 5, 6]
                            }, 

                            "lambda_l1": {
                                "values": [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e1]
                            },
                            "lambda_l2": {
                                "values": [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e1]
                            },
                            "min_child_weight": {
                                "values": [1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1e1]
                            }
                         }

    else :
                params = {
                            "learning_rate": {
                                "values": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
                            },
                            "random_strength": {
                                "values": [2, 4, 6, 8, 10, 12, 14, 16, 18, 19]
                            },
                            "one_hot_max_size": {
                                "values": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
                            },
                            "l2_leaf_reg": {
                                "values": [2, 3, 4, 5, 6, 7, 8, 9]
                            },  
                            "bagging_temperature": {
                                "values": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                            },
                            "leaf_estimation_iterations": {
                                "values": [2, 4, 6, 8, 10, 12, 14, 16, 18, 19]
                            }
                }

            
    sweep_config = {
        "method": method_name, # try bayes,grid or random
        "metric": {
                "name": "accuracy",
                "goal": "maximize"   
         },
         "parameters" : params
    }

    sweep_id = wandb.sweep(sweep_config, project=algorithm_name+"_"+dataset_name+"_"+method_name)


    
    # Set up your default hyperparameters
    if dataset_name == "airlines":
        with open('./conf/dataset/airlines.yaml',"r") as file:
            cfg = yaml.safe_load(file)
    if dataset_name == "eye_movement":
        with open('./conf/dataset/eye_movement.yaml',"r") as file:
            cfg = yaml.safe_load(file)
    if dataset_name == "gas-concentration":
        with open('./conf/dataset/gas-concentration.yaml',"r") as file:
            cfg = yaml.safe_load(file)
    if dataset_name == "gesture-phase":
        with open('./conf/dataset/gesture-phase.yaml',"r") as file:
            cfg = yaml.safe_load(file)
            
    dataset_path = cfg['dataset']['path']
    dataset_target_col = cfg['dataset']['target_col']
 
    
    #-------------------read data-----------------------------#
    dataframe = pd.read_csv(dataset_path)
    
    #------------------data preprocessing--------------------#
    if dataset_name == "airlines":
        categ = ['Airline','AirportFrom','AirportTo']
        le = LabelEncoder()
        dataframe[categ] = dataframe[categ].apply(le.fit_transform)
    if algorithm_name == "LGBMClassifier":
        new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in dataframe.columns}
        new_n_list = list(new_names.values())
        new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
        dataframe = dataframe.rename(columns=new_names)
        
    X = dataframe.drop(columns=[dataset_target_col])
    target = getattr(dataframe, dataset_target_col)
    y = target
    
    #for converting categorical to labels 
    if not dataset_name == "eye_movement":
        le = LabelEncoder()
        y = le.fit_transform(y)
    #------------------data preprocessing--------------------#


    #--------splitting dataset into train,validation,test sets 80,20-----#
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=dataframe[dataset_target_col])
    
    #-----------------model_selection---------------------------# 
    def classifier_model(algorithm, dataset, param): 
        if algorithm == "LGBMClassifier": 
            model = lightgbm.LGBMClassifier(**param)
        elif algorithm == "XGBClassifier":
            model = xgboost.XGBClassifier(**param)
        else:
            if dataset == "eye_movement":
                model = catboost.CatBoostClassifier(**param, thread_count=40, cat_features=["P1stFixation", "P2stFixation", "nextWordRegress"],  verbose=None)
            elif dataset == "airlines":
                model = catboost.CatBoostClassifier(**param, thread_count=40, cat_features=['Airline','AirportFrom','AirportTo'])
            else:
                model = catboost.CatBoostClassifier(**param, thread_count=40)
        return model 
 

    if algorithm_name == "LGBMClassifier":
        if dataset_name != "airlines":
            params['objective'] = "multiclass"
            if dataset_name == "eye_movement":
                params["num_class"] = 4
            elif dataset_name == "gas-concentration":
                params["num_class"] = 6
            else:
                params["num_class"] = 9
        else:
            params['objective'] = "binary"
    
    #for iteration 
    index_list = [0]
    iteration_list = []
    score_list = []    
    
    def train():
        if algorithm_name == "XGBClassifier":
            config_defaults = {
            "alpha": 1e-16,
            "colsample_bylevel": 0.2,
            "colsample_bytree": 0.9,
            "eta": 1e-05,
            "gamma": 1e-14,
            "lambda": 1e-14,
            "max_depth": 5,
            "min_child_weight": 0,
            "n_estimators": 1100,
            "subsample": 0.9,
          }
            
        elif algorithm_name == "LGBMClassifier":
            config_defaults = {
                'class_weight': 'balanced',
                'boosting_type': 'dart',
                'num_leaves': 60,
                'learning_rate': 0.02,
                'subsample_for_bin': 240000,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.5,
                'min_data_in_leaf': 3,
                'lambda_l1': 1e-4,
                'lambda_l2': 1e-6,
                'min_child_weight': 1e-1,
                'verbose': -1,
            } 
        else:           
            config_defaults = {
                'learning_rate': 1e-5,
                'random_strength': 11,
                'one_hot_max_size': 10,
                'l2_leaf_reg': 5,
                'bagging_temperature': 0.5,
                'leaf_estimation_iterations': 2
            }
            
        wandb.init(project=algorithm_name+""+dataset_name+"_"+method_name, config=config_defaults);       
        config = wandb.config       
                   
        #splitting dataset into train_validation,test sets 80,20
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
        model = classifier_model(algorithm_name, dataset_name, config)
        params={'verbose': False} 
        scores = cross_validate(model, X_train_val, y_train_val, cv=5, scoring='accuracy', fit_params=params)
        cv_accuracy = np.mean(scores['test_score'])
        wandb.log({"CV_Accuracy": cv_accuracy})
        
        model.fit(X_train_val, y_train_val, verbose=False)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        
        wandb.log({"Accuracy": accuracy}) 
                    
        iteration_list.append(index_list[-1])
        score_list.append(accuracy)
        index_list.append(index_list[-1] +1)
        
        wandb.log({"Max Accuracy": max(score_list)})
        
    wandb.agent(sweep_id, train, count=30)
        
if __name__ == "__main__":
    main()
    
    
    
    

