#!/bin/bash
#
# Run train
#


ENV_TYPE=dqn_agent_equity_HU
MODEL_NAME=dqn4
CALL_EQUITY=0.5
BET_EQUITY=0.7
NB_STEPS=400000
NB_MAX_START_STEPS=400
NB_STEPS_WARMUP=600


python train.py \
    --env_type $ENV_TYPE \
    --model_name $MODEL_NAME \
    --call_equity $CALL_EQUITY \
    --bet_equity $BET_EQUITY \
    --nb_steps $NB_STEPS \
    --nb_max_start_steps $NB_MAX_START_STEPS \
    --nb_steps_warmup $NB_STEPS_WARMUP
