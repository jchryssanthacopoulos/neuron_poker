"""Player based on a trained neural network."""

import logging
import json
import time
from typing import Optional

import numpy as np
from rl.agents import DQNAgent
from rl.core import Processor
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam

from agents import PlayerBase
from gym_env.action import Action
from gym_env.logger import PlayLogger
from gym_env.processor import LegalMovesProcessor


log = logging.getLogger(__name__)


class Player(PlayerBase):
    """Mandatory class with the player methods."""

    def __init__(self, load_model=None, env=None, nb_steps=400000, nb_max_start_steps=400, nb_steps_warmup=600):
        """Initialize the agent.

        Args:
            load_model: Name of model to load
            env: Name of environment to load
            nb_steps: Number of steps to simulate in training
            nb_max_start_steps: Maximum number of random steps to take at the beginning
            nb_steps_warmup: Number of warmup steps to take

        """
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.autoplay = True

        self.dqn = None
        self.model = None
        self.env = env

        # algorithm hyperparameters
        self.nb_steps = nb_steps
        self.nb_max_start_steps = nb_max_start_steps
        self.nb_steps_warmup = nb_steps_warmup
        self.window_length = 1
        self.memory_limit = int(self.nb_steps / 5)
        self.train_interval = 100      # train every X steps
        self.batch_size = 500          # items sampled from memory to train
        self.enable_double_dqn = False

        if load_model:
            self.load(load_model)

    def initiate_agent(self, env):
        """Initiate a deep Q agent."""
        tf.compat.v1.disable_eager_execution()

        self.env = env

        nb_actions = self.env.action_space.n

        self.model = Sequential()
        self.model.add(Dense(512, activation='relu', input_shape=env.observation_space.shape))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(nb_actions, activation='linear'))

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=self.memory_limit, window_length=self.window_length)
        policy = TrumpPolicy()

        nb_actions = env.action_space.n

        self.dqn = DQNAgent(
            model=self.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=self.nb_steps_warmup,
            target_model_update=1e-2, policy=policy, processor=LegalMovesProcessor(env.num_opponents, nb_actions),
            batch_size=self.batch_size, train_interval=self.train_interval, enable_double_dqn=self.enable_double_dqn
        )
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    def start_step_policy(self, observation):
        """Custom policy for random decisions for warm up."""
        log.info("Random action")
        _ = observation
        action = self.env.action_space.sample()
        return action

    def train(self, env_name):
        """Train a model."""
        # initiate training loop
        timestr = time.strftime("%Y%m%d-%H%M%S") + "_" + str(env_name)
        tensorboard = TensorBoard(
            log_dir='./Graph/{}'.format(timestr), histogram_freq=0, write_graph=True, write_images=False
        )

        self.dqn.fit(
            self.env, nb_max_start_steps=self.nb_max_start_steps, nb_steps=self.nb_steps, visualize=False, verbose=2,
            start_step_policy=self.start_step_policy, callbacks=[tensorboard]
        )

        # Save the architecture
        dqn_json = self.model.to_json()
        with open("models/dqn_{}_json.json".format(env_name), "w") as json_file:
            json.dump(dqn_json, json_file)

        # After training is done, we save the final weights.
        self.dqn.save_weights('models/dqn_{}_weights.h5'.format(env_name), overwrite=True)

        # Finally, evaluate our algorithm for 5 episodes
        self.dqn.test(self.env, nb_episodes=5, visualize=False)

    def load(self, env_name):
        """Load a model."""

        # Load the architecture
        with open('models/dqn_{}_json.json'.format(env_name), 'r') as architecture_json:
            dqn_json = json.load(architecture_json)

        self.model = model_from_json(dqn_json)
        self.model.load_weights('models/dqn_{}_weights.h5'.format(env_name))

    def play(self, nb_episodes: int = 5, render: bool = False, processor: Optional[Processor] = None) -> PlayLogger:
        """Let the agent play.

        Args:
            nb_episodes: Number of episodes to run
            render: Whether to render
            processor: Process to use to process actions, observations, etc.

        """
        nb_actions = self.env.action_space.n

        if processor is None:
            processor = LegalMovesProcessor(self.env.num_opponents, nb_actions)

        memory = SequentialMemory(limit=self.memory_limit, window_length=self.window_length)
        policy = TrumpPolicy()

        self.dqn = DQNAgent(
            model=self.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=self.nb_steps_warmup,
            target_model_update=1e-2, policy=policy, processor=processor, batch_size=self.batch_size,
            train_interval=self.train_interval, enable_double_dqn=self.enable_double_dqn
        )
        self.dqn.compile(Adam(lr=1e-3), metrics=['mae'])  # pylint: disable=no-member

        play_logger = PlayLogger()
        play_logger._set_env(self.env)

        self.dqn.test(self.env, nb_episodes=nb_episodes, visualize=render, callbacks=[play_logger])

        # return history of player stacks and actions
        return play_logger

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info

        this_player_action_space = {
            Action.FOLD,
            Action.CHECK,
            Action.CALL,
            Action.RAISE_POT,
            Action.ALL_IN
        }

        _ = this_player_action_space.intersection(set(action_space))

        action = None
        return action


class TrumpPolicy(BoltzmannQPolicy):
    """Custom policy when making decision based on neural network."""

    def select_action(self, q_values: np.array):
        """Return the selected action.

        Arguments
            q_values: List of the estimations of Q for each action

        Returns
            Selection action

        """
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        log.info(f"Chosen action by keras-rl {action} - probabilities: {probs}")

        return action
