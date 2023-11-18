"""Random player."""

import random

from agents import PlayerBase
from gym_env.env_jc import Action


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
