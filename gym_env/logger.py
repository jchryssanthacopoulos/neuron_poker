"""Different types of play loggers."""

import pandas as pd
from rl.callbacks import Callback


class PlayLogger(Callback):
    """Logger of player stats per hand."""

    def __init__(self):
        self.stacks = pd.DataFrame()
        self.actions = pd.DataFrame()
        self.winner_in_hands = pd.DataFrame()
        self.winner_in_episodes = pd.DataFrame()

    def on_episode_end(self, episode, logs={}):
        """Called at end of each episode."""
        stacks_history = self.env.funds_history.copy(deep=True)
        stacks_history['episode'] = episode + 1
        self.stacks = pd.concat([self.stacks, stacks_history])

        actions_history = self.env.actions_history.copy(deep=True)
        actions_history['episode'] = episode + 1
        self.actions = pd.concat([self.actions, actions_history])

        self.winner_in_hands = pd.concat([self.winner_in_hands, pd.DataFrame.from_dict({'winner': self.env.winner_in_hands})])
        self.winner_in_episodes = pd.concat([self.winner_in_episodes, pd.DataFrame.from_dict({'winner': self.env.winner_in_episodes})])
