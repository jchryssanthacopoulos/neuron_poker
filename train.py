""""Script to train AI agent."""

import argparse
import logging

import gym
import numpy as np

from agents.agent_consider_equity import Player as EquityPlayer
from agents.agent_keras_rl_dqn import Player as DQNPlayer
from gym_env.env import PlayerShell
from tools.helper import get_config
from tools.helper import init_logger


ENV_NAME = 'neuron_poker-v0'


class Trainer:
    """Class to train agents."""

    def __init__(
            self,
            stack: float,
            small_blind: float,
            big_blind: float,
            render: bool,
            use_cpp_montecarlo: bool
    ):
        """Initialize."""
        self.stack = stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.render = render
        self.use_cpp_montecarlo = use_cpp_montecarlo
        self.log = logging.getLogger(__name__)

    def dqn_train_equity_HU(
            self,
            model_name: str,
            call_equity: float,
            bet_equity: float,
            nb_steps: int,
            nb_max_start_steps: int,
            nb_steps_warmup: int
    ):
        """Train a DQN model against an equity player in heads-up play.

        Args:
            model_name: Name of model to save
            call_equity: Minimum equity for equity player to call
            bet_equity: Minimum equity for equity player to bet
            nb_steps: Number of steps to simulate in training
            nb_max_start_steps: Maximum number of random steps to take at the beginning
            nb_steps_warmup: Number of warmup steps to take

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
            use_cpp_montecarlo=self.use_cpp_montecarlo
        )

        np.random.seed(123)
        env.reset(seed=123)

        dqn = DQNPlayer(nb_steps=nb_steps, nb_max_start_steps=nb_max_start_steps, nb_steps_warmup=nb_steps_warmup)
        dqn.initiate_agent(env)
        dqn.train(model_name)


if __name__ == '__main__':
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_type", help="Environment to run", choices=['dqn_agent_equity_HU'], type=str, required=True)
    parser.add_argument("--model_name", help="DQN agent model name", type=str, default='dqn1')
    parser.add_argument("--stack", help="Initial stack", type=float, default=500)
    parser.add_argument("--small_blind", help="Small blind", type=float, default=2.5)
    parser.add_argument("--big_blind", help="Big blind", type=float, default=5)
    parser.add_argument("--call_equity", help="Minimum equity to call of equity player", type=float, default=0.5)
    parser.add_argument("--bet_equity", help="Minimum equity to bet of equity player", type=float, default=0.7)
    parser.add_argument("--nb_steps", help="Number of steps to simulate", type=int, default=4e5)
    parser.add_argument("--nb_max_start_steps", help="Maximum number of random steps to take at the beginning", type=int, default=400)
    parser.add_argument("--nb_steps_warmup", help="Number of warmup steps to take", type=int, default=600)
    parser.add_argument("--log_file", help="Log file", type=str, default='default')
    parser.add_argument("--log_level", help="Log level", type=int, default=logging.INFO)
    parser.add_argument("--render", help="Whether to render", action="store_true", default=False)
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

    trainer = Trainer(
        stack=args.stack,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        render=args.render,
        use_cpp_montecarlo=args.use_cpp_montecarlo
    )

    # this is the only option
    trainer.dqn_train_equity_HU(
        args.model_name,
        args.call_equity,
        args.bet_equity,
        args.nb_steps,
        args.nb_max_start_steps,
        args.nb_steps_warmup
    )
