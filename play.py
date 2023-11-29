"""Script to simulate play and collect statistics."""

import argparse
from copy import deepcopy
import logging

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents.agent_consider_equity import Player as EquityPlayer
from agents.agent_keras_rl_dqn import Player as DQNPlayer
from agents.agent_random import Player as RandomPlayer
from gym_env.action import ActionBlind
from gym_env.env import PlayerShell
from gym_env.logger import PlayLogger
from gym_env.processor import SingleOpponentLegalMovesProcessor
from tools.helper import get_config
from tools.helper import init_logger


ENV_NAME = 'neuron_poker-v0'


class SelfPlay:
    """Class to simulate play between players."""

    def __init__(
            self,
            num_episodes: int,
            stack: float,
            small_blind: float,
            big_blind: float,
            render: bool,
            use_cpp_montecarlo: bool
    ):
        """Initialize."""
        self.num_episodes = num_episodes
        self.stack = stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.render = render
        self.use_cpp_montecarlo = use_cpp_montecarlo
        self.log = logging.getLogger(__name__)

        self.bot_space = [
            EquityPlayer(name='equity/10/30', min_call_equity=0.1, min_bet_equity=0.3),
            EquityPlayer(name='equity/20/40', min_call_equity=0.2, min_bet_equity=0.4),
            EquityPlayer(name='equity/30/50', min_call_equity=0.3, min_bet_equity=0.5),
            EquityPlayer(name='equity/40/60', min_call_equity=0.4, min_bet_equity=0.6),
            EquityPlayer(name='equity/50/70', min_call_equity=0.5, min_bet_equity=0.7),
            RandomPlayer(),
            RandomPlayer(),
            RandomPlayer()
        ]

    def random_vs_equity_heads_up(self, call_equity: float, bet_equity: float) -> PlayLogger:
        """Simulate heads-up play between random and equity players.

        Args:
            call_equity: Minimum equity for equity player to call
            bet_equity: Minimum equity for equity player to bet

        """
        player = RandomPlayer()
        bots = [
            EquityPlayer(
                name=f'equity/{call_equity}/{bet_equity}', min_call_equity=call_equity, min_bet_equity=bet_equity
            )
        ]

        env = gym.make(
            ENV_NAME,
            player=player,
            bots=bots,
            initial_stacks=self.stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            render=self.render,
            use_cpp_montecarlo=self.use_cpp_montecarlo,
            terminate_if_main_player_lost=False
        )

        # used to collect statistics
        play_logger = PlayLogger()
        play_logger._set_env(env)

        for i in range(self.num_episodes):
            env.reset()
            play_logger.on_episode_end(i)

        return play_logger

    def dqn_agent_vs_equity_heads_up(self, model_name: str, call_equity: float, bet_equity: float) -> PlayLogger:
        """Simulate heads-up play between DQN agent and equity players.

        Args:
            model_name: Name of model to load
            call_equity: Minimum equity for equity player to call
            bet_equity: Minimum equity for equity player to bet

        """
        player = PlayerShell(name='keras-rl', stack_size=self.stack)
        bots = [
            EquityPlayer(
                name=f'equity/{call_equity}/{bet_equity}', min_call_equity=call_equity, min_bet_equity=bet_equity
            )
        ]

        env = gym.make(
            ENV_NAME,
            player=player,
            bots=bots,
            initial_stacks=self.stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            render=self.render,
            use_cpp_montecarlo=self.use_cpp_montecarlo,
            terminate_if_main_player_lost=False
        )

        np.random.seed(123)
        env.reset(seed=123)

        dqn = DQNPlayer(load_model=model_name, env=env)
        play_logger = dqn.play(nb_episodes=self.num_episodes, render=self.render)

        return play_logger

    def dqn_agent_vs_five_players(self, model_name: str, randomize_bots: bool = False) -> PlayLogger:
        """Simulate play between DQN agent and five other players.

        Args:
            model_name: Name of model to load
            randomize_bots: Whether to select random bots to play against

        """
        player = PlayerShell(name='keras-rl', stack_size=self.stack)

        if randomize_bots:
            rand_idx = np.random.randint(len(self.bot_space), size=5)
            bots = []
            for idx in rand_idx:
                bots.append(deepcopy(self.bot_space[idx]))
        else:
            bots = [
                EquityPlayer(name='equity/10/30', min_call_equity=0.1, min_bet_equity=0.3),
                RandomPlayer(),
                EquityPlayer(name='equity/30/50', min_call_equity=0.3, min_bet_equity=0.5),
                RandomPlayer(),
                EquityPlayer(name='equity/50/70', min_call_equity=0.5, min_bet_equity=0.7)
            ]

        env = gym.make(
            ENV_NAME,
            player=player,
            bots=bots,
            initial_stacks=self.stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            render=self.render,
            use_cpp_montecarlo=self.use_cpp_montecarlo,
            terminate_if_main_player_lost=False
        )

        # reduce observation space down to one opponent
        # processor = SingleOpponentLegalMovesProcessor(env.num_opponents, env.action_space.n)

        np.random.seed(123)
        env.reset(seed=123)

        dqn = DQNPlayer(load_model=model_name, env=env)
        play_logger = dqn.play(nb_episodes=self.num_episodes, render=self.render)

        return play_logger


def display_stats(play_logger: PlayLogger):
    """Display player stats.

    Args:
        play_logger: Play logger with stacks, actions, etc.

    """
    num_episodes = play_logger.stacks.episode.max()

    # get cumulative wins/losses across hands
    diff_stacks = pd.DataFrame()
    for i in range(num_episodes):
        stacks_for_episode = play_logger.stacks[play_logger.stacks.episode == i + 1]
        diff_stacks = pd.concat([diff_stacks, stacks_for_episode.diff().dropna().drop('episode', axis=1)])

    # plot change in stacks
    plt.figure(figsize=(7, 5))
    diff_stacks.cumsum().plot(use_index=False, ax=plt.gca())

    # plot players' actions
    for i in range(play_logger.env.num_of_players):
        player_actions = play_logger.actions[play_logger.actions.player == i].action

        # remove small and big blind
        player_actions = player_actions.apply(
            lambda x: x.name if x not in [ActionBlind.SMALL, ActionBlind.BIG] else None
        )

        plt.figure(figsize=(7, 5))
        player_actions.hist()
        plt.title(f"Player {i} actions")

    plt.figure(figsize=(7, 5))
    play_logger.winner_in_hands.winner.hist()
    plt.title("Winners of hands")

    plt.figure(figsize=(7, 5))
    play_logger.winner_in_episodes.winner.hist()
    plt.title("Winners of episodes")

    plt.show()


if __name__ == '__main__':
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_type",
        help="Environment to run",
        choices=['random_equity_HU', 'dqn_agent_equity_HU', 'dqn_agent_five_players'],
        type=str,
        required=True
    )
    parser.add_argument("--model_name", help="DQN agent model name", type=str, default='dqn1')
    parser.add_argument("--num_episodes", help="Number of episodes to simulate", type=int, default=1)
    parser.add_argument("--stack", help="Initial stack", type=float, default=500)
    parser.add_argument("--small_blind", help="Small blind", type=float, default=2.5)
    parser.add_argument("--big_blind", help="Big blind", type=float, default=5)
    parser.add_argument("--call_equity", help="Minimum equity to call of equity player", type=float, default=0.5)
    parser.add_argument("--bet_equity", help="Minimum equity to bet of equity player", type=float, default=0.7)
    parser.add_argument("--log_file", help="Log file", type=str, default='default')
    parser.add_argument("--log_level", help="Log level", type=int, default=logging.INFO)
    parser.add_argument("--render", help="Whether to render", action="store_true", default=False)
    parser.add_argument("--randomize_bots", help="Whether to randomize opponents", action="store_true", default=False)
    parser.add_argument(
        "--use_cpp_montecarlo",
        help="Whether to use CPP file for MC simulation",
        action="store_true",
        default=True,
        required=False
    )
    args = parser.parse_args()

    _ = get_config()
    init_logger(screenlevel=args.log_level, filename=args.log_file)

    runner = SelfPlay(
        render=args.render,
        num_episodes=args.num_episodes,
        use_cpp_montecarlo=args.use_cpp_montecarlo,
        stack=args.stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind
    )

    if args.env_type == 'random_equity_HU':
        play_logger = runner.random_vs_equity_heads_up(args.call_equity, args.bet_equity)
    elif args.env_type == 'dqn_agent_equity_HU':
        play_logger = runner.dqn_agent_vs_equity_heads_up(args.model_name, args.call_equity, args.bet_equity)
    else:
        play_logger = runner.dqn_agent_vs_five_players(args.model_name, args.randomize_bots)

    display_stats(play_logger)
