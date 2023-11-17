"""Define the observation space."""

from typing import List

import numpy as np

from tools.helper import flatten


class CommunityData:
    """Data available to everybody"""

    def __init__(self, num_opponents: int):
        """Initialize."""
        self.stage = [False] * 4  # one hot: preflop, flop, turn, river
        self.community_pot = 0
        self.current_round_pot = 0
        self.stacks = [0] * num_opponents
        self.blinds = [0] * num_opponents

    def ndim(self):
        """Get number of dimensions."""
        return len(list(flatten(self.__dict__.values())))


class StageData:
    """Preflop, flop, turn and river"""

    def __init__(self, num_players: int):
        """Initialize."""
        self.calls = [False] * num_players  # ix[0] = dealer
        self.raises = [False] * num_players  # ix[0] = dealer
        self.min_call_at_action = [0] * num_players  # ix[0] = dealer
        self.contribution = [0] * num_players  # ix[0] = dealer
        self.stack_at_action = [0] * num_players  # ix[0] = dealer
        self.community_pot_at_action = [0] * num_players  # ix[0] = dealer

    def ndim(self):
        """Get number of dimensions."""
        return len(list(flatten(self.__dict__.values())))


class PlayerData:
    "Player specific information"

    def __init__(self, num_actions: int):
        """Initialize."""
        self.stack = 0
        self.is_dealer = False
        self.blind = 0
        self.equity_to_river_alive = 0
        self.amount_to_call = 0
        self.legal_moves = [0] * num_actions

    def ndim(self):
        """Get number of dimensions."""
        return len(list(flatten(self.__dict__.values())))


class Observation:
    """Class to encapsulate all the observations."""

    def __init__(self, player_data: PlayerData, community_data: CommunityData, stage_data: List[StageData]):
        """Initialize."""
        self.player_data = player_data
        self.community_data = community_data
        # self.stage_data = stage_data

    def ndim(self):
        """Get number of dimensions."""
        n = self.player_data.ndim()
        n += self.community_data.ndim()
        # n += len(self.stage_data) * self.stage_data[0].ndim()
        return n

    def to_array(self):
        """Convert to numpy array."""
        arr1 = np.array(list(flatten(self.player_data.__dict__.values())), dtype='float32')
        arr2 = np.array(list(flatten(self.community_data.__dict__.values())), dtype='float32')
        # arr3 = np.array([list(flatten(sd.__dict__.values())) for sd in self.stage_data], dtype='float32').flatten()
        return np.concatenate([arr1, arr2]).flatten()

    def __str__(self):
        """Get print-friendly version of the observation."""
        s = "Player data:\n"
        for k, v in self.player_data.__dict__.items():
            s += f"  {k}: {v}\n"

        s += "Community data:\n"
        for k, v in self.community_data.__dict__.items():
            s += f"  {k}: {v}\n"

        # s += "Stage data:\n"
        # for i in range(len(self.stage_data)):
        #     s += f"  Round {i + 1}\n"
        #     for k, v in self.stage_data[i].__dict__.items():
        #         s += f"    {k}: {v}\n"

        return s[:-1]
