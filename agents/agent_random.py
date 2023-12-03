"""Random player."""

import random

import numpy as np

from agents import PlayerBase
from gym_env.action import Action


class Player(PlayerBase):
    """Mandatory class with the player methods."""

    def __init__(self, name='Random'):
        """Initiaization of an agent."""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info

        this_player_action_space = [
            Action.FOLD,
            Action.CHECK,
            Action.CALL,
            Action.RAISE_POT,
            Action.ALL_IN
        ]

        possible_moves = []
        for action in this_player_action_space:
            if action in action_space:
                possible_moves.append(action)

        action = random.choice(possible_moves)

        return action


class RandomPlayer2(PlayerBase):
    """Mandatory class with the player methods."""

    def __init__(self, name: str = 'Random', all_in_prob: float = 0.025):
        """Initiaization of an agent.

        Args:
            name: Agent name
            all_in_prob: Probability of going all-in

        """
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.all_in_prob = all_in_prob
        self.autoplay = True

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info

        if Action.ALL_IN in action_space:
            # if all in is an available action, choose it with probability all_in_prob
            sampl = np.random.uniform(low=0, high=1, size=1)[0]
            if sampl < self.all_in_prob:
                return Action.ALL_IN

        this_player_action_space = [
            Action.FOLD,
            Action.CHECK,
            Action.CALL,
            Action.RAISE_POT
        ]

        possible_moves = []
        for action in this_player_action_space:
            if action in action_space:
                possible_moves.append(action)

        action = random.choice(possible_moves)

        return action
