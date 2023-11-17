"""
Neuron poker

Usage:
  main.py selfplay random [options]
  main.py selfplay keypress [options]
  main.py selfplay consider_equity [options]
  main.py selfplay equity_improvement --improvement_rounds=<> [options]
  main.py selfplay dqn_train [options]
  main.py selfplay dqn_play [options]
  main.py learn_table_scraping [options]

options:
  -h --help                 Show this screen.
  -r --render               render screen
  -c --use_cpp_montecarlo   use cpp implementation of equity calculator. Requires cpp compiler but is 500x faster
  -f --funds_plot           Plot funds at end of episode
  --log                     log file
  --name=<>                 Name of the saved model
  --screenloglevel=<>       log level on screen
  --episodes=<>             number of episodes to play
  --stack=<>                starting stack for each player [default: 500]
  --small_blind=<>          Small blind [default: 5]
  --big_blind=<>            Big blind [default: 10]

"""

import logging

from docopt import docopt
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agents.agent_consider_equity import Player as EquityPlayer
from agents.agent_custom_q1 import Player as Custom_Q1
from agents.agent_keypress import Player as KeyPressAgent
from agents.agent_keras_rl_dqn import Player as DQNPlayer
from agents.agent_random import Player as RandomPlayer
from gym_env.env import PlayerShell
from tools.helper import get_config
from tools.helper import init_logger


# pylint: disable=import-outside-toplevel

def command_line_parser():
    """Entry function."""
    args = docopt(__doc__)

    if args['--log']:
        logfile = args['--log']
    else:
        print("Using default log file")
        logfile = 'default'

    model_name = args['--name'] if args['--name'] else 'dqn1'
    screenloglevel = logging.INFO if not args['--screenloglevel'] else \
        getattr(logging, args['--screenloglevel'].upper())

    _ = get_config()

    init_logger(screenlevel=screenloglevel, filename=logfile)
    print(f"Screenloglevel: {screenloglevel}")

    log = logging.getLogger("")
    log.info("Initializing program")

    if args['selfplay']:
        num_episodes = 1 if not args['--episodes'] else int(args['--episodes'])
        runner = SelfPlay(
            render=args['--render'],
            num_episodes=num_episodes,
            use_cpp_montecarlo=args['--use_cpp_montecarlo'],
            funds_plot=args['--funds_plot'],
            stack=int(args['--stack']),
            small_blind=int(args['--small_blind']),
            big_blind=int(args['--big_blind'])
        )

        if args['random']:
            runner.random_agents()

        elif args['keypress']:
            runner.key_press_agents()

        elif args['consider_equity']:
            runner.equity_vs_random()

        elif args['equity_improvement']:
            improvement_rounds = int(args['--improvement_rounds'])
            runner.equity_self_improvement(improvement_rounds)

        elif args['dqn_train']:
            runner.dqn_train_keras_rl(model_name)

        elif args['dqn_play']:
            runner.dqn_play_keras_rl(model_name)
    else:
        raise RuntimeError("Argument not yet implemented")


class SelfPlay:
    """Orchestration of playing against itself."""

    def __init__(self, render, num_episodes, use_cpp_montecarlo, funds_plot, stack=500, small_blind=5, big_blind=10):
        """Initialize."""
        self.winner_in_episodes = []
        self.use_cpp_montecarlo = use_cpp_montecarlo
        self.funds_plot = funds_plot
        self.render = render
        self.env = None
        self.num_episodes = num_episodes
        self.stack = stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.log = logging.getLogger(__name__)

    def random_agents(self):
        """Create an environment with 6 random players."""
        env_name = 'neuron_poker-v0'
        num_of_plrs = 2
        self.env = gym.make(
            env_name,
            initial_stacks=self.stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            render=self.render
        )
        for _ in range(num_of_plrs):
            player = RandomPlayer()
            self.env.add_player(player)

        self.env.reset()

    def key_press_agents(self):
        """Create an environment with 6 key press agents."""
        env_name = 'neuron_poker-v0'
        num_of_plrs = 2
        self.env = gym.make(
            env_name,
            initial_stacks=self.stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            render=self.render
        )
        for _ in range(num_of_plrs):
            player = KeyPressAgent()
            self.env.add_player(player)

        self.env.reset()

    def equity_vs_random(self):
        """Create 6 players, 4 of them equity based, 2 of them random."""
        env_name = 'neuron_poker-v0'

        self.env = gym.make(
            env_name,
            initial_stacks=self.stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            render=self.render
        )
        self.env.add_player(EquityPlayer(name='equity/50/50', min_call_equity=.5, min_bet_equity=-.5))
        self.env.add_player(EquityPlayer(name='equity/50/80', min_call_equity=.8, min_bet_equity=-.8))
        self.env.add_player(EquityPlayer(name='equity/70/70', min_call_equity=.7, min_bet_equity=-.7))
        self.env.add_player(EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=-.3))
        self.env.add_player(RandomPlayer())
        self.env.add_player(RandomPlayer())

        for _ in range(self.num_episodes):
            self.env.reset()
            self.winner_in_episodes.append(self.env.winner_ix)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print(league_table)
        print(f"Best Player: {best_player}")

    def equity_self_improvement(self, improvement_rounds):
        """Create 6 players, 4 of them equity based, 2 of them random."""
        calling = [.1, .2, .3, .4, .5, .6]
        betting = [.2, .3, .4, .5, .6, .7]

        for improvement_round in range(improvement_rounds):
            env_name = 'neuron_poker-v0'
            self.env = gym.make(
                env_name,
                initial_stacks=self.stack,
                small_blind=self.small_blind,
                big_blind=self.big_blind,
                render=self.render
            )
            for i in range(6):
                self.env.add_player(EquityPlayer(name=f'Equity/{calling[i]}/{betting[i]}',
                                                 min_call_equity=calling[i],
                                                 min_bet_equity=betting[i]))

            for _ in range(self.num_episodes):
                self.env.reset()
                self.winner_in_episodes.append(self.env.winner_ix)

            league_table = pd.Series(self.winner_in_episodes).value_counts()
            best_player = int(league_table.index[0])
            print(league_table)
            print(f"Best Player: {best_player}")

            # self improve
            self.log.info(f"Self improvment round {improvement_round}")
            for i in range(6):
                calling[i] = np.mean([calling[i], calling[best_player]])
                self.log.info(f"New calling for player {i} is {calling[i]}")
                betting[i] = np.mean([betting[i], betting[best_player]])
                self.log.info(f"New betting for player {i} is {betting[i]}")

    def dqn_train_keras_rl(self, model_name):
        """Implementation of kreras-rl deep q learing."""
        env_name = 'neuron_poker-v0'

        player = PlayerShell(name='keras-rl', stack_size=self.stack)
        bots = [
            EquityPlayer(name='equity/50/70', min_call_equity=0.5, min_bet_equity=0.7)
        ]
        env = gym.make(
            env_name,
            player=player,
            bots=bots,
            initial_stacks=self.stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            funds_plot=self.funds_plot,
            render=self.render,
            use_cpp_montecarlo=self.use_cpp_montecarlo,
            check_fold_on_illegal_move=True
        )

        np.random.seed(123)
        env.reset(seed=123)

        dqn = DQNPlayer()
        dqn.initiate_agent(env)
        dqn.train(env_name=model_name)

    def dqn_play_keras_rl(self, model_name):
        """Create 6 players, one of them a trained DQN."""
        env_name = 'neuron_poker-v0'

        player = PlayerShell(name='keras-rl', stack_size=self.stack)
        bots = [
            EquityPlayer(name='equity/50/50', min_call_equity=0.5, min_bet_equity=0.7),
            EquityPlayer(name='equity/50/80', min_call_equity=0.2, min_bet_equity=0.3),
            EquityPlayer(name='equity/70/70', min_call_equity=0.7, min_bet_equity=0.7),
            EquityPlayer(name='equity/20/30', min_call_equity=0.2, min_bet_equity=0.3),
            RandomPlayer()
        ]
        self.env = gym.make(
            env_name,
            player=player,
            bots=bots,
            initial_stacks=self.stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            render=self.render,
            use_cpp_montecarlo=self.use_cpp_montecarlo,
            check_fold_on_illegal_move=True,  # needed to prevent agent from getting stuck on illegal moves
            terminate_if_main_player_lost=False  # continue to simulate after main player has no funds
        )

        np.random.seed(123)
        self.env.reset(seed=123)

        dqn = DQNPlayer(load_model=model_name, env=self.env)
        stacks = dqn.play(nb_episodes=self.num_episodes, render=self.render)

        # get cumulative wins/losses across hands
        diff_stacks = pd.DataFrame()

        for i in range(self.num_episodes):
            stacks_for_episode = stacks[stacks.episode == i + 1]
            diff_stacks = pd.concat([diff_stacks, stacks_for_episode.diff().dropna().drop('episode', axis=1)])

        diff_stacks.cumsum().plot(use_index=False)
        plt.show()

    def dqn_train_custom_q1(self):
        """Create 6 players, 4 of them equity based, 2 of them random."""
        env_name = 'neuron_poker-v0'

        player = Custom_Q1(name='Deep_Q1')
        bots = [
            EquityPlayer(name='equity/20/30', min_call_equity=.2, min_bet_equity=-.3),
            RandomPlayer(),
            RandomPlayer()
        ]

        self.env = gym.make(
            env_name,
            player=player,
            bots=bots,
            initial_stacks=self.stack,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            render=self.render
        )

        for _ in range(self.num_episodes):
            self.env.reset()
            self.winner_in_episodes.append(self.env.winner_ix)

        league_table = pd.Series(self.winner_in_episodes).value_counts()
        best_player = league_table.index[0]

        print("League Table")
        print("============")
        print(league_table)
        print(f"Best Player: {best_player}")


if __name__ == '__main__':
    command_line_parser()
