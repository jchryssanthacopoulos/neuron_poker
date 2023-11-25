#!/bin/bash
#
# Run play
#
# Equity players to consider:
#   CALL_EQUITY = 0.1, BET_EQUITY = 0.3
#   CALL_EQUITY = 0.3, BET_EQUITY = 0.5
#   CALL_EQUITY = 0.5, BET_EQUITY = 0.7
#   CALL_EQUITY = 0.7, BET_EQUITY = 0.9
#


ENV_TYPE=random_equity_HU
NUM_EPISODES=500
MODEL_NAME=dqn3
CALL_EQUITY=0.7
BET_EQUITY=0.9


python play.py \
    --env_type $ENV_TYPE \
    --num_episodes $NUM_EPISODES \
    --model_name $MODEL_NAME \
    --call_equity $CALL_EQUITY \
    --bet_equity $BET_EQUITY
