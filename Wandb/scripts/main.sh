#!/bin/sh

##SBATCH --gpus-per-node=1
#SBATCH --account=pls0151
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task 40
#SBATCH -J diffneighb
##SBATCH -o sputha -%j.out #replace
#SBATCH --mail-type=END
#SBATCH --mail-user=sputha@student.ysu.edu #replace


module load miniconda3/4.10.3-py37
source activate hpo #replace
#source activate autoTPred #replace
## srun python ./eye_movement.py callbacks=default datamodule.loader="neighbor" datamodule.neighb=5 > eye_movement_output.txt

# srun python ./Hyperopt/random/Catboost/eye_movement_random.py callbacks=default datamodule.loader="neighbor" datamodule.neighb=5 > ./Hyperopt/random/Catboost/eye_movement_random_output.txt

# srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=gas-concentration sampler=tpe algorithm=catboost > ./gas-concentration-hyperopt_tpe.txt &

# srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=gas-concentration sampler=rand algorithm=catboost > ./gas-concentration-hyperopt_rand.txt &

# srun python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=gas-concentration sampler=rand,tpe algorithm=xgboost,catboost > ./gas-concentration-rest.txt


# python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=eye_movement sampler=tpe algorithm=catboost,xgboost > ./justin.txt
# srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=gas-concentration sampler=tpe algorithm=catboost > ./gas_new_2.txt

##LGBM Algorithm
#srun python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=eye_movement sampler=rand algorithm=lightgbm > ./eye_movement_optuna_rand.txt
#srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=eye_movement sampler=tpe algorithm=lightgbm > ./eye_movement_hyperopt_tpe.txt
#srun python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=eye_movement sampler=tpe algorithm=lightgbm > ./eye_movement_opt_tpe.txt
#srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=eye_movement sampler=rand algorithm=lightgbm > ./eye_movement_hyperopt_rand.txt 

#srun python ./xgboost_w.py --dataset gas-concentration --algorithm XGBClassifier > ./gas-concentration_random_wandb.txt
#srun python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=gas-concentration sampler=rand algorithm=lightgbm > ./gas-concentration_rand_optuna.txt 
#srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=gas-concentration sampler=rand algorithm=lightgbm > ./gas-concentration_rand_hyperopt.txt 
#srun python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=gas-concentration sampler=tpe algorithm=lightgbm > ./gas-concentration_tpe_optuna.txt 
#srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=gas-concentration sampler=tpe algorithm=lightgbm > ./gas-concentration_tpe_hyperopt.txt 


#srun python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=airlines sampler=rand algorithm=lightgbm > ./airlines_rand_optuna.txt 
#srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=airlines sampler=rand algorithm=lightgbm > ./airlines_rand_hyperopt.txt 
#srun python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=airlines sampler=tpe algorithm=lightgbm > ./airlines_tpe_optuna.txt 
#srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=airlines sampler=tpe algorithm=lightgbm > ./airlines_tpe_hyperopt.txt 

#srun python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=gesture-phase sampler=rand algorithm=lightgbm > ./gesture-phase_rand_optuna.txt 
#srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=gesture-phase sampler=rand algorithm=lightgbm > ./gesture-phase_rand_hyperopt.txt
#srun python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=gesture-phase sampler=tpe algorithm=lightgbm > ./gesture-phase_tpe_optuna.txt 
#srun python ./my_app.py --framework hyperopt --dataset gesture-phase --sampler rand --algorithm LGBMClassifier > ./gesture-phase_rand_hyperopt2.txt
#srun python ./my_app.py --framework optuna --dataset gesture-phase --sampler tpe --algorithm LGBMClassifier > ./gesture-phase_tpe_optuna1.txt
#srun python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=eye_movement sampler=tpe algorithm=lightgbm > ./eye_movement_lgbm2_wand.txt


#------------------wandb--------------------#


#srun python ./xgboost_w.py --dataset gesture-phase --algorithm XGBClassifier --sweep_method random> ./gesture-phase_xg_random_wandb.txt
#srun python ./xgboost_w.py --dataset gesture-phase --algorithm XGBClassifier --sweep_method bayes> ./gesture-phase_xg_bayes_wandb.txt
#srun python ./xgboost_w.py --dataset gesture-phase --algorithm XGBClassifier > ./gesture-phase_xg_grid_wandb.txt

#srun python ./catboost_w.py --dataset gesture-phase --algorithm CatBoostClassifier --sweep_method random > ./gesture-phase_cat_random_wandb.txt
#srun python ./catboost_w.py --dataset gesture-phase --algorithm CatBoostClassifier --sweep_method bayes > ./gesture-phase_cat_bayes_wandb.txt
#srun python ./catboost_w.py --dataset gesture-phase --algorithm CatBoostClassifier > ./gesture-phase_cat_grid_wandb.txt

#srun python ./lgboost_w.py --dataset gesture-phase --algorithm LGBMClassifier --sweep_method random > ./gesture-phase_lgt_random_wandb.txt
#srun python ./lgboost_w.py --dataset gesture-phase --algorithm LGBMClassifier --sweep_method bayes > ./gesture-phase_lgt_bayes_wandb.txt
#srun python ./lgboost_w.py --dataset gesture-phase --algorithm LGBMClassifier --sweep_method grid > ./gesture-phase_lgt_grid_wandb.txt


#srun python ./xgboost_w.py --dataset gas-concentration --algorithm XGBClassifier --sweep_method random > ./gas-concentration_xg_random_wandb.txt
#srun python ./xgboost_w.py --dataset gas-concentration --algorithm XGBClassifier --sweep_method bayes > ./gas-concentration_xg_bayes_wandb.txt
#srun python ./xgboost_w.py --dataset gas-concentration --algorithm XGBClassifier > ./gas-concentration_xg_grid_wandb.txt

#srun python ./catboost_w.py --dataset gas-concentration --algorithm CatBoostClassifier --sweep_method random > ./gas-concentration_cat_random_wandb.txt
#srun python ./catboost_w.py --dataset gas-concentration --algorithm CatBoostClassifier --sweep_method bayes > ./gas-concentration_cat_bayes_wandb.txt
#srun python ./catboost_w.py --dataset gas-concentration --algorithm CatBoostClassifier --sweep_method grid > ./gas-concentration_cat_grid_wandb.txt

#srun python ./lgboost_w.py --dataset gas-concentration --algorithm LGBMClassifier --sweep_method random > ./gas-concentration_lgt_random_wandb.txt
#srun python ./xgboost_w.py --dataset gas-concentration --algorithm LGBMClassifier --sweep_method bayes > ./gas-concentration_lgt_bayes_wandb.txt
#srun python ./lgboost_w.py --dataset gas-concentration --algorithm LGBMClassifier > ./gas-concentration_lgt_grid_wandb.txt



#srun python ./xgboost_w.py --dataset airlines --algorithm XGBClassifier --sweep_method random > ./airlines_xg_random_wandb.txt
#srun python ./xgboost_w.py --dataset airlines --algorithm XGBClassifier --sweep_method bayes > ./airlines_xg_bayes_wandb.txt
#srun python ./xgboost_w.py --dataset airlines --algorithm XGBClassifier > ./airlines_xg_grid_wandb.txt

#srun python ./catboost_w.py --dataset airlines --algorithm CatBoostClassifier --sweep_method random > ./airlines_cat_random_wandb.txt
#srun python ./catboost_w.py --dataset airlines --algorithm CatBoostClassifier --sweep_method bayes > ./airlines_cat_bayes_wandb.txt
#srun python ./catboost_w.py --dataset airlines --algorithm CatBoostClassifier > ./airlines_cat_grid_wandb.txt

#srun python ./lgboost_w.py --dataset airlines --algorithm LGBMClassifier --sweep_method random > ./airlines_lgt_random_wandb.txt
#srun python ./lgboost_w.py --dataset airlines --algorithm LGBMClassifier --sweep_method bayes > ./airlines_lgt_bayes_wandb.txt
#srun python ./lgboost_w.py --dataset airlines --algorithm LGBMClassifier > ./airlines_lgt_grid_wandb.txt


srun python ./xgboost_w.py --dataset eye_movement --algorithm XGBClassifier --sweep_method random > ./eye_movement_xg_random_wandb.txt
#srun python ./xgboost_w.py --dataset eye_movement --algorithm XGBClassifier --sweep_method bayes > ./eye_movement_xg_bayes_wandb.txt
#srun python ./xgboost_w.py --dataset eye_movement --algorithm XGBClassifier > ./eye_movement_xg_grid_wandb.txt

#srun python ./catboost_w.py --dataset eye_movement --algorithm CatBoostClassifier --sweep_method random > ./eye_movement_cat_random_wandb.txt
#srun python ./catboost_w.py --dataset eye_movement --algorithm CatBoostClassifier --sweep_method bayes > ./eye_movement_cat_bayes_wandb.txt
#srun python ./catboost_w.py --dataset eye_movement --algorithm CatBoostClassifier > ./eye_movement_cat_grid_wandb.txt

#srun python ./lgboost_w.py --dataset eye_movement --algorithm LGBMClassifier --sweep_method random > ./eye_movement_lgt_random_wandb.txt
#srun python ./lgboost_w.py --dataset eye_movement --algorithm LGBMClassifier --sweep_method bayes > ./eye_movement_lgt_bayes_wandb.txt
#srun python ./lgboost_w.py --dataset eye_movement --algorithm LGBMClassifier > ./eye_movement_lgt_grid_wandb.txt
