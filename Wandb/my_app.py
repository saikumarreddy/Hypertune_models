#libraries 
import hydra
import logging
from omegaconf import DictConfig
import os 

import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.plotting import main_plot_history
import optuna
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.samplers import RandomSampler,TPESampler

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

# A logger for this file
log = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
 
    log.info(cfg.algorithm.name + "_" + cfg.dataset.name + "_" + cfg.framework.name + "_" + cfg.sampler.name )
    #read data
    dataframe = pd.read_csv(cfg.dataset.path)
    
    #data preprocessing 
    if cfg.dataset.name == "airlines":
        categ = ['Airline','AirportFrom','AirportTo']
        le = LabelEncoder()
        dataframe[categ] = dataframe[categ].apply(le.fit_transform)
    if cfg.algorithm.name == "LGBMClassifier":
        new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in dataframe.columns}
        new_n_list = list(new_names.values())
        new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
        dataframe = dataframe.rename(columns=new_names)
        
    X = dataframe.drop(columns=[cfg.dataset.target_col])
    target = getattr(dataframe, cfg.dataset.target_col)
    y = target
    
    #for converting categorical to labels 
    if not cfg.dataset.name == "eye_movement":
        le = LabelEncoder()
        y = le.fit_transform(y)
        
    max_evals = 30
    
    #for iteration output csv 
    index_list = [0]
    iteration_list = []
    score_list = []
    
    #splitting dataset into train,validation,test sets 80,10,20
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=dataframe[cfg.dataset.target_col])
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val)
    
    #returns model 
    def classifier_model(algorithm, dataset, param):
        if algorithm == "CatBoostClassifier":
            if dataset == "eye_movement":
                model = catboost.CatBoostClassifier(**param, thread_count=40, cat_features=["P1stFixation", "P2stFixation", "nextWordRegress"],  verbose=None)
            elif dataset == "airlines":
                model = catboost.CatBoostClassifier(**param, thread_count=40, cat_features=['Airline','AirportFrom','AirportTo'])
            else:
                model = catboost.CatBoostClassifier(**param, thread_count=40)
                
        elif algorithm == "LGBMClassifier": 
            model = lightgbm.LGBMClassifier(**param)
        else: 
            model = xgboost.XGBClassifier(**param)
        return model 
 
    
    #HYPEROPT
    if(cfg.framework.name == "hyperopt"): 
        if(cfg.algorithm.name == "CatBoostClassifier"):
            param = {
                'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1)),
                'random_strength': hp.quniform('random_strength', 1, 20,1),
                'one_hot_max_size': hp.quniform('one_hot_max_size', 0,25,1),
                'l2_leaf_reg': hp.loguniform('l2_leaf_reg', np.log(1), np.log(10)),
                'bagging_temperature': hp.uniform('bagging_temperature', 0, 1),
                'leaf_estimation_iterations': hp.quniform('leaf_estimation_iterations', 1, 20, 1)
            }
        elif(cfg.algorithm.name == "XGBClassifier"):
            param = {
                'n_estimators': hp.randint('n_estimators', 100, 4000),
                'eta': hp.loguniform('eta', np.log(1e-7), np.log(1)),
                'max_depth': hp.randint('max_depth', 1, 10),
                'subsample': hp.uniform('subsample', 0.2, 1),
                'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
                'colsample_bylevel': hp.uniform('colsample_bylevel', 0.2, 1),
                'min_child_weight': hp.loguniform('min_child_weight', np.log(1e-16), np.log(1e5)),
                'alpha': hp.loguniform('alpha_log', np.log(1e-16), np.log(1e1)),
                'lambda': hp.loguniform('lambda_log', np.log(1e-16), np.log(1e1)),
                'gamma': hp.loguniform('gamma_log', np.log(1e-16), np.log(1e1))
            }
            
        else:
            param = {
                'class_weight': hp.choice('class_weight', [None, 'balanced']),
                'boosting_type': hp.choice('boosting_type',['gbdt','dart','goss']),
                'num_leaves': hp.randint('num_leaves', 30, 150),
                'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
                'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
                'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1), #alias "subsample"
                'min_data_in_leaf': hp.randint('min_data_in_leaf', 0, 6),
                'lambda_l1': hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)]),
                'lambda_l2': hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)]),
                'verbose': -1,
                #the following not being used due to other params, so trying to silence the complaints by setting to None
                'subsample': None, #overridden by bagging_fraction
                'reg_alpha': None, #overridden by lambda_l1
                'reg_lambda': None, #overridden by lambda_l2
                'min_sum_hessian_in_leaf': None, #overrides min_child_weight
                'min_child_samples': None, #overridden by min_data_in_leaf
                'colsample_bytree': None, #overridden by feature_fraction
                'min_child_weight': hp.loguniform('min_child_weight', -16, 5), #also aliases to min_sum_hessian
            }
            
            if cfg.algorithm.name == "LGBMClassifier":
                if cfg.dataset.name != "airlines":
                    param['objective'] = "multiclass"
                    if cfg.dataset.name == "eye_movement":
                        param["num_class"] = 4
                    elif cfg.dataset.name == "gas-concentration":
                        param["num_class"] = 6
                    else:
                        param["num_class"] = 9
                else:
                    param['objective'] = "binary"

        def hyperparameter_tuning(param):
           
            model = classifier_model(cfg.algorithm.name, cfg.dataset.name, param)
            params={'verbose': False} 
            scores = cross_validate(model, X_train_val, y_train_val, cv=5, scoring='accuracy', fit_params=params)
            accuracy = np.mean(scores['test_score'])
            log.info("-----validation accuracy----")
            log.info(accuracy)
            
            iteration_list.append(index_list[-1])
            score_list.append(accuracy)
            index_list.append(index_list[-1] +1)
            return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}
        
        
        algoithm = getattr(hyperopt, cfg.sampler.name)
        
        t0 = time.time()
        trials = Trials()
        best = fmin(fn=hyperparameter_tuning,
                    algo=algoithm.suggest,
                    space=param,
                    max_evals=max_evals,
                    trials=trials
                   )
        
#         accuracy_scores.append(min(trials.losses()))
#         timings.append(time_taken_total/60)
        best_params = hyperopt.space_eval(param, best)
        best_loss = trials.best_trial['result']['loss']
        #best_accuracy = trials.best_trial['result']['accuracy']
  
        time_taken_total = time.time()-t0
        log.info("---TOTAL TIME TAKEN--- = ")
        log.info(time_taken_total/60)

        log.info("------ BEST VALUE -------")
        log.info(min(trials.losses()))
        
        
        log.info("-----PARAMS-----")
        best_params = hyperopt.space_eval(param, best)
        log.info(hyperopt.space_eval(param, best))
        
        model = classifier_model(cfg.algorithm.name, cfg.dataset.name, best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        
        log.info("-----Test accuray-----")
        log.info(accuracy)
        log.info("--------jobname---------")
        log.info( cfg.dataset.name + "_" + cfg.algorithm.nickname + "_" + cfg.framework.name + "_"  + cfg.sampler.name)
        log.info("-----------------------------END---------------------------------")
        
    #OPTUNA
    else:
        # wandb.init(project="Optuna",entity="saikumarreddy1101",reinit=True,)
        wandb_kwargs = {"project": "optuna1","entity": "saikumarreddy1101","reinit": True,}
        wandbc = WeightsAndBiasesCallback(metric_name="accuracy", wandb_kwargs=wandb_kwargs,as_multirun=True)
        
        @wandbc.track_in_wandb() #decorator to log wandb inside objective function
        def objective(trial):
            if(cfg.algorithm.name == "CatBoostClassifier"):

                param = {
                    'learning_rate': trial.suggest_loguniform('learning_rate',1e-5, 1),
                    'random_strength': trial.suggest_discrete_uniform('random_strength', 1, 20,1),
                    'one_hot_max_size': trial.suggest_discrete_uniform('one_hot_max_size', 0,25,1),
                    'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1, 10),
                    'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0, 1),
                    'leaf_estimation_iterations': trial.suggest_discrete_uniform('leaf_estimation_iterations', 1, 20, 1)
                }
            elif(cfg.algorithm.name == "XGBClassifier"):
                param = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 4000),
                    'eta': trial.suggest_loguniform('eta', 1e-7, 1),
                    'max_depth': trial.suggest_int('max_depth', 1, 10, 1),
                    'subsample': trial.suggest_uniform('subsample', 0.2, 1),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.2, 1),
                    'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.2, 1),
                    'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-16, 1e5),
                    'alpha': trial.suggest_loguniform('alpha', 1e-16, 1e1),
                    'lambda': trial.suggest_loguniform('lambda', 1e-16, 1e1),
                    'gamma': trial.suggest_loguniform('gamma', 1e-16, 1e1),
                }
                    
            else:
                param = {
                    'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                    'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
                    'num_leaves': trial.suggest_int('num_leaves', 30, 150),
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
                    'subsample_for_bin': trial.suggest_int('subsample_for_bin', 20000, 300000, step=20000),
                    'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1),
                    'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 0, 6),
                    'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-16, 1e2) ,
                    'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-16, 1e2) ,
                    'min_child_weight': trial.suggest_loguniform('min_child_weight', 1e-16, 1e2),
                    'verbose': -1,
                }
                
            if cfg.algorithm.name == "LGBMClassifier":
                if cfg.dataset.name != "airlines":
                    param['objective'] = "multiclass"

                    if cfg.dataset.name == "eye_movement":
                        param["num_class"] = 4
                    elif cfg.dataset.name == "gas-concentration":
                        param["num_class"] = 6
                    else:
                        param["num_class"] = 9
                else:
                    param['objective'] = "binary"
   
                
            model = classifier_model(cfg.algorithm.name, cfg.dataset.name, param)
            params={'verbose': False} 
            scores = cross_validate(model, X_train_val, y_train_val, cv=5, scoring='accuracy', fit_params=params)
            accuracy = np.mean(scores['test_score'])
            log.info("-----validation accuracy----")
            log.info(accuracy)
            
            wandb.log({"accuracy": accuracy})
            
            iteration_list.append(index_list[-1])
            score_list.append(accuracy)
            index_list.append(index_list[-1] +1)
            return accuracy
          

        
        t0 = time.time()
        if(cfg.sampler.name == "tpe"):
            study = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=100, reduction_factor=3))
        else: 
            study = optuna.create_study(direction='maximize', sampler=RandomSampler(), pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=30, interval_steps=10
                ))
        study.optimize(objective, n_trials=max_evals)

        time_taken_total = time.time()-t0
        log.info("---TOTAL TIME TAKEN--- = ")
        log.info(time_taken_total/60)
        
        log.info("---- BEST VALUE ----- ")
        log.info(study.best_value)
        log.info(" --- PARAMS -----")
        best_params = study.best_params
        log.info(study.best_params)
        
        f = "best_{}".format
        for param_name, param_value in study.best_trial.params.items():
            wandb.run.summary[f(param_name)] = param_value

        wandb.run.summary["best accuracy"] = study.best_trial.value
        wandb.log(
            {
                "optuna_optimization_history": optuna.visualization.plot_optimization_history(
                    study
                ),
                "optuna_param_importances": optuna.visualization.plot_param_importances(
                    study
                ),
            }
            
        )    
        # Finish Wandb run
        wandb.finish()
        
        model = classifier_model(cfg.algorithm.name, cfg.dataset.name, best_params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        
        log.info("-----Test accuray-----")
        log.info(accuracy)
        log.info("--------jobname---------")
        log.info( cfg.dataset.name + "_" + cfg.algorithm.nickname + "_" + cfg.framework.name + "_"  + cfg.sampler.name)
        
        log.info("-----------------------------END---------------------------------")

        
    iter_score = pd.DataFrame({'iteration': iteration_list, 'Accuracy': score_list})
    iter_score = iter_score.reset_index(drop=True)
    path = "/users/PLS0151/sputha/Hypertune/plots"
    iter_score.to_csv(os.path.join(path, cfg.dataset.name +  "_" + cfg.algorithm.nickname + "_" + cfg.framework.name + "_" + cfg.sampler.name + ".csv"), index=False)

if __name__ == "__main__":
    main()
    
    
    
    

