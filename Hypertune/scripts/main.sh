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

# srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=gas-concentration sampler=rand algorithm=catboost > ./gas-concentration-rest-4.txt

# srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=gas-concentration sampler=rand,tpe algorithm=xgboost,catboost > ./gas-concentration-rest.txt


# python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=eye_movement sampler=tpe algorithm=catboost,xgboost > ./justin.txt
# srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=gas-concentration sampler=tpe algorithm=catboost > ./gas_new_2.txt

##LGBM Algorithm
#srun python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=eye_movement sampler=rand,tpe algorithm=lightgbm > ./eye_movement_lgbm.txt
#srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=eye_movement sampler=rand,tpe algorithm=lightgbm > ./eye_movement_lgbm1.txt

#srun python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=gas-concentration sampler=rand,tpe algorithm=lightgbm > ./gas-concentration_lgbm_optuna.txt &
#srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=gas-concentration sampler=rand,tpe algorithm=lightgbm > ./gas-concentration_lgbm_hyperopt.txt &

#srun python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=airlines sampler=rand,tpe algorithm=lightgbm > ./airlines_lgbm_optuna.txt &
#srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=airlines sampler=rand,tpe algorithm=lightgbm > ./airlines_lgbm_hyperopt.txt &

#srun python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=gesture-phase sampler=rand,tpe algorithm=lightgbm > ./gesture-phase_lgbm_optuna.txt &
#srun python ./my_app.py hydra.mode=MULTIRUN framework=hyperopt dataset=gesture-phase sampler=rand,tpe algorithm=lightgbm > ./gesture-phase_lgbm_hyperopt.txt

srun python ./my_app.py hydra.mode=MULTIRUN framework=optuna dataset=eye_movement sampler=tpe algorithm=lightgbm > ./eye_movement_lgbm1_wand.txt
