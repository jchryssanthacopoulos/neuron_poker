"""Define poker environment."""

from enum import Enum
import logging
import time
from typing import Dict
from typing import List
from typing import Optional

from gym import Env
from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np
import pandas as pd

from agents import PlayerBase
from gym_env.action import Action
from gym_env.action import ActionBlind
from gym_env.observation import Observation
from gym_env.observation import StageData
from gym_env.rendering import PygletWindow, WHITE, RED, GREEN, BLUE
from tools.hand_evaluator import get_winner

# pylint: disable=import-outside-toplevel

log = logging.getLogger(__name__)

winner_in_episodes = []


class Stage(Enum):
    """Different game stages."""

    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    END_HIDDEN = 4
    SHOWDOWN = 5


class HoldemTable(Env):
    """Pokergame environment."""

    def __init__(
            self,
            player: PlayerBase,
            bots: List[PlayerBase],
            initial_stacks: int = 500,
            small_blind: int = 2.5,
            big_blind: int = 5,
            render: bool = False,
            funds_plot: bool = True,
            max_raising_rounds: int = 2,
            use_cpp_montecarlo: bool = False,
            check_fold_on_illegal_move: bool = False,
            terminate_if_main_player_lost: bool = True,
            normalize_pot_values: bool = True
    ):
        """The table needs to be initialized once at the beginning.

        Args:
            player: Main player
            bots: Bots at the table
            initial_stacks: initial stacks per placyer
            small_blind: Value of small blind
            big_blind: Value of big blind
            render: render table after each move in graphical format
            funds_plot: show plot of funds history at end of each episode
            max_raising_rounds: max raises per round per player
            use_cpp_montecarlo: Whether to use C++ version of Monte Carlo simulator
            check_fold_on_illegal_move: Whether to resort to check/fold if action is illegal
            terminate_if_main_player_lost: Whether to end the game if the main player has no more funds
            normalize_pot_values: Whether to normalize pot values by big blind when saving history

        """
        if use_cpp_montecarlo:
            import cppimport
            calculator = cppimport.imp("tools.montecarlo_cpp.pymontecarlo")
            get_equity = calculator.montecarlo
        else:
            from tools.montecarlo_python import get_equity

        self.num_of_players = 0
        self.get_equity = get_equity
        self.use_cpp_montecarlo = use_cpp_montecarlo
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.render_switch = render
        self.players = []
        self.table_cards = None
        self.dealer_pos = None
        self.player_status = []  # one hot encoded
        self.current_player = None
        self.player_cycle = None  # cycle iterator
        self.stage = None
        self.last_player_pot = None
        self.viewer = None
        self.player_max_win = None  # used for side pots
        self.second_round = False
        self.last_caller = None
        self.last_raiser = None
        self.raisers = []
        self.callers = []
        self.played_in_round = None
        self.min_call = None
        self.deck = None
        self.action = None
        self.winner_ix = None
        self.initial_stacks = initial_stacks
        self.funds_plot = funds_plot
        self.max_round_raising = max_raising_rounds
        self.check_fold_on_illegal_move = check_fold_on_illegal_move
        self.terminate_if_main_player_lost = terminate_if_main_player_lost

        # pots
        self.community_pot = 0
        self.current_round_pot = 9
        self.player_pots = None  # individual player pots

        self.info = None
        self.done = False
        self.early_stop = False
        self.funds_history = None
        self.legal_moves = None
        self.illegal_move_reward = -100
        self.action_space = Discrete(len(Action))
        self.first_action_for_hand = None

        self.initiated_new_hand = False
        self.num_hands_session = 0
        self.total_num_hands = 0
        self.max_num_of_hands = 1000
        self.time_in_hand = None

        self.pot_norm = 100 * self.big_blind if normalize_pot_values else 1

        # add players to table, starting with the main player
        self.add_player(player)
        for bot in bots:
            self.add_player(bot)

        self.num_opponents = self.num_of_players - 1
        self.observation = Observation(self.num_opponents, len(Action))
        self.observation_space = Box(low=np.zeros(self.observation.ndim()), high=np.ones(self.observation.ndim()) * 100)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset after game over."""
        super().reset(seed=seed)

        log.info("Start of reset")

        self.observation = Observation(self.num_opponents, len(Action))
        self.info = None
        self.done = False
        self.early_stop = False
        self.funds_history = pd.DataFrame()
        self.first_action_for_hand = [True] * len(self.players)

        for player in self.players:
            player.stack = self.initial_stacks

        self.dealer_pos = 0
        self.player_cycle = PlayerCycle(
            self.players,
            dealer_idx=-1,
            max_steps_after_raiser=len(self.players) - 1,
            max_steps_after_big_blind=len(self.players)
        )
        self._start_new_hand()
        self._get_environment()

        # auto play for agents where autoplay is set
        self._advance_autoplay_players()

        observation = self.observation.to_array()

        log.info(f"Observation =\n{str(self.observation)}")
        log.info(f"Observation (array version) = {observation}")
        log.info("End of reset")

        return observation

    def step(self, action):  # pylint: disable=arguments-differ
        """Next player makes a move and a new environment is observed.

        Args:
            action: Used for testing only. Needs to be of Action type

        """
        log.info("Start of step")
        log.info(f"Action: {action}")

        # this is not an autoplay agent, meaning the action has to be used
        if self._agent_is_autoplay():
            raise RuntimeError("Agent to step is autoplay agent.")

        if Action(action) not in self.legal_moves:
            raise RuntimeError(f"Action {action} is illegal from current state")

        self._get_environment()

        should_step = True
        if should_step:
            self._execute_step(Action(action))
            self._advance_autoplay_players()
            reward = self._calculate_reward(action)

        observation = self.observation.to_array()

        log.info(f"Observation =\n{str(self.observation)}")
        log.info(f"Observation (array version) = {observation}")
        log.info(f"End of step. Reward = {reward}")

        return observation, reward, self.done, self.info

    def _advance_autoplay_players(self):
        """Advance all autoplay players one round."""
        while self._agent_is_autoplay() and not self.done:
            # call agent's action method
            log.info("Autoplay agent. Call action method of agent.")
            action = self.current_player.agent_obj.action(
                self.legal_moves, self.observation.to_array(), self.info
            )

            self._execute_step(Action(action))

    def _execute_step(self, action):
        self._process_decision(action)

        self._next_player()

        if self.stage in [Stage.END_HIDDEN, Stage.SHOWDOWN]:
            self._end_hand()
            self._start_new_hand()

        self._get_environment()

    def _agent_is_autoplay(self, idx=None):
        if not idx:
            return hasattr(self.current_player.agent_obj, 'autoplay')
        return hasattr(self.players[idx].agent_obj, 'autoplay')

    def _get_environment(self):
        """Observe the environment."""
        if not self.done:
            self._get_legal_moves()

        # set community data
        self.observation.community_data.set(
            self.players[1:],
            self.community_pot,
            self.current_round_pot,
            self.stage.value,
            self.small_blind,
            self.big_blind,
            self.pot_norm
        )

        if not self.current_player:  # game over
            # get equity of main player
            self.current_player = self.players[0]

        # set player data
        amount_to_call = (
            min(self.min_call - self.player_pots[self.players[0].seat], self.players[0].stack) / self.pot_norm
        )
        equity = self.get_equity(
            set(self.current_player.cards), set(self.table_cards), sum(self.player_cycle.alive), 1000
        )
        self.observation.player_data.set(
            self.players[0],
            self.dealer_pos,
            self.legal_moves,
            amount_to_call,
            equity,
            self.small_blind,
            self.big_blind,
            self.pot_norm
        )

        log.info(
            f"Computed equity for player {self.current_player.seat} with cards "
            f"{self.current_player.cards} (table cards: {self.table_cards}): {self.observation.player_data.data['equity']}"
        )

        # save equity
        self.current_player.equity_alive = self.observation.player_data.data['equity']

        self._get_legal_moves()

        self.info = {
            'player_data': self.observation.player_data.data,
            'community_data': self.observation.community_data,
            'stage_data': [stage.data for stage in self.observation.stage_data],
            'legal_moves': self.legal_moves
        }

        if self.render_switch:
            self.render()

    def _calculate_reward(self, last_action):
        """Preliminiary implementation of reward function.

        Currently missing potential additional winnings from future contributions

        """
        # if last_action == Action.FOLD:
        #     reward = -(self.community_pot + self.current_round_pot)
        # else:
        #     reward = self.player_data.equity_to_river_alive * (self.community_pot + self.current_round_pot) - \
        #         (1 - self.player_data.equity_to_river_alive) * self.player_pots[self.current_player.seat]

        _ = last_action

        if self.done and not self.early_stop:
            won = 1 if not self._agent_is_autoplay(idx=self.winner_ix) else -1
            reward = self.initial_stacks * len(self.players) * won
            return reward

        if len(self.funds_history) > 1 and (self.initiated_new_hand or self.early_stop):
            self.initiated_new_hand = False
            return self.funds_history.iloc[-1, 0] - self.funds_history.iloc[-2, 0]

        return 0

    def _process_decision(self, action):  # pylint: disable=too-many-statements
        """Process the decisions that have been made by an agent."""
        if action not in [ActionBlind.SMALL, ActionBlind.BIG]:
            assert action in set(self.legal_moves), "Illegal decision"

        if action == Action.FOLD:
            self.player_cycle.deactivate_current()
            self.player_cycle.mark_folder()
        else:
            if action == Action.CALL:
                contribution = min(
                    self.min_call - self.player_pots[self.current_player.seat],
                    self.current_player.stack
                )
                self.callers.append(self.current_player.seat)
                self.last_caller = self.current_player.seat

            # verify the player has enough in his stack
            elif action == Action.CHECK:
                contribution = 0
                self.player_cycle.mark_checker()

            elif action == Action.RAISE_POT:
                contribution = (self.community_pot + self.current_round_pot)
                self.raisers.append(self.current_player.seat)

            elif action == Action.ALL_IN:
                contribution = self.current_player.stack
                self.raisers.append(self.current_player.seat)

            elif action == ActionBlind.SMALL:
                contribution = np.minimum(self.small_blind, self.current_player.stack)

            elif action == ActionBlind.BIG:
                contribution = np.minimum(self.big_blind, self.current_player.stack)
                self.player_cycle.mark_bb()
            else:
                raise RuntimeError("Illegal action.")

            if contribution > self.min_call:
                self.player_cycle.mark_raiser()

            self.current_player.stack -= contribution
            self.player_pots[self.current_player.seat] += contribution
            self.current_round_pot += contribution
            self.last_player_pot = self.player_pots[self.current_player.seat]

            if self.current_player.stack == 0 and contribution > 0:
                self.player_cycle.mark_out_of_cash_but_contributed()

            self.min_call = max(self.player_pots)

            self.current_player.actions.append(action)
            self.current_player.last_action_in_stage = action.name
            self.current_player.temp_stack.append(self.current_player.stack)

            self.player_max_win[self.current_player.seat] += contribution  # side pot

            # set stage data
            pos = self.player_cycle.idx
            rnd = self.stage.value + self.second_round
            self.observation.stage_data[rnd].set(
                pos,
                self.current_player.stack,
                action,
                self.min_call,
                self.community_pot,
                contribution,
                self.pot_norm
            )

        self.player_cycle.update_alive()

        log.info(
            f"Seat {self.current_player.seat} ({self.current_player.name}): {action} - Remaining stack: {self.current_player.stack}, "
            f"Round pot: {self.current_round_pot}, Community pot: {self.community_pot}, "
            f"player pot: {self.player_pots[self.current_player.seat]}"
        )

    def _start_new_hand(self):
        """Deal new cards to players and reset table states."""
        self._save_funds_history()

        if self._check_game_over():
            return

        if self.time_in_hand is not None:
            log.info(f"Time in hand: {time.time() - self.time_in_hand}")

        log.info("")
        log.info("+++++++++++++++++++")
        log.info(f"Starting hand {self.total_num_hands + 1}")
        log.info("+++++++++++++++++++")

        self.time_in_hand = time.time()
        self.num_hands_session += 1
        self.total_num_hands += 1
        self.initiated_new_hand = True
        self.table_cards = []
        self._create_card_deck()
        self.stage = Stage.PREFLOP

        self.observation = Observation(self.num_opponents, len(Action))

        # pots
        self.community_pot = 0
        self.current_round_pot = 0
        self.player_pots = [0] * len(self.players)
        self.player_max_win = [0] * len(self.players)
        self.last_player_pot = 0
        self.played_in_round = 0
        self.first_action_for_hand = [True] * len(self.players)

        for player in self.players:
            player.cards = []
            player.actions = []

        self._next_dealer()

        self._distribute_cards()
        self._initiate_round()

    def _save_funds_history(self):
        """Keep track of player funds history."""
        funds_dict = {i: player.stack for i, player in enumerate(self.players)}
        self.funds_history = pd.concat([self.funds_history, pd.DataFrame(funds_dict, index=[0])])

    def _check_game_over(self):
        """Check if only one player has money left."""
        player_alive = []
        self.player_cycle.new_hand_reset()

        log.info("Check game over")

        for idx, player in enumerate(self.players):
            if player.stack > 0:
                player_alive.append(True)
            else:
                self.player_status.append(False)
                self.player_cycle.deactivate_player(idx)

        if self.terminate_if_main_player_lost and not self.player_cycle.can_still_make_moves_in_this_hand[0]:
            # if main player cannot play, end round
            log.info("Main player lost.")
            self._game_over()
            return True

        if self.num_hands_session and self.num_hands_session % self.max_num_of_hands == 0:
            log.info("Maximum number of hands reached in session.")
            self.early_stop = True
            self._game_over()
            return True

        remaining_players = sum(player_alive)
        if remaining_players < 2:
            self._game_over()
            return True
        return False

    def _game_over(self):
        """End of an episode."""
        log.info("Game over.")
        self.num_hands_session = 0
        self.done = True
        player_names = [f"{i} - {player.name}" for i, player in enumerate(self.players)]
        self.funds_history.columns = player_names
        log.info(self.funds_history)

        winner_in_episodes.append(self.winner_ix)
        league_table = pd.Series(winner_in_episodes).value_counts()
        best_player = league_table.index[0]
        log.info(league_table)
        log.info(f"Best Player: {best_player}")

    def _initiate_round(self):
        """A new round (flop, turn, river) is initiated."""
        self.last_caller = None
        self.last_raiser = None
        self.raisers = []
        self.callers = []
        self.min_call = 0
        for player in self.players:
            player.last_action_in_stage = ''
        self.player_cycle.new_round_reset()

        if self.stage == Stage.PREFLOP:
            log.info("")
            log.info("===Round: Stage: PREFLOP")
            # max steps total will be adjusted again at bb
            self.player_cycle.max_steps_total = len(self.players) * self.max_round_raising + 2

            self._next_player()
            self._process_decision(ActionBlind.SMALL)
            self._next_player()
            self._process_decision(ActionBlind.BIG)
            self._next_player()

        elif self.stage in [Stage.FLOP, Stage.TURN, Stage.RIVER]:
            self.player_cycle.max_steps_total = len(self.players) * self.max_round_raising

            self._next_player()

        elif self.stage == Stage.SHOWDOWN:
            log.info("Showdown")

        else:
            raise RuntimeError()

    def add_player(self, agent):
        """Add a player to the table. Has to happen at the very beginning."""
        self.num_of_players += 1
        player = PlayerShell(stack_size=self.initial_stacks, name=agent.name)
        player.agent_obj = agent
        player.seat = len(self.players)  # assign next seat number to player
        player.stack = self.initial_stacks
        self.players.append(player)
        self.player_status = [True] * len(self.players)
        self.player_pots = [0] * len(self.players)

    def _end_round(self):
        """End of preflop, flop, turn or river."""
        self._close_round()
        if self.stage == Stage.PREFLOP:
            self.stage = Stage.FLOP
            self._distribute_cards_to_table(3)

        elif self.stage == Stage.FLOP:
            self.stage = Stage.TURN
            self._distribute_cards_to_table(1)

        elif self.stage == Stage.TURN:
            self.stage = Stage.RIVER
            self._distribute_cards_to_table(1)

        elif self.stage == Stage.RIVER:
            self.stage = Stage.SHOWDOWN

        log.info("--------------------------------")
        log.info(f"===ROUND: {self.stage} ===")
        self._clean_up_pots()

    def _clean_up_pots(self):
        self.community_pot += self.current_round_pot
        self.current_round_pot = 0
        self.player_pots = [0] * len(self.players)

    def _end_hand(self):
        self._clean_up_pots()
        self.winner_ix = self._get_winner()
        self._award_winner(self.winner_ix)

    def _get_winner(self):
        """Determine which player has won the hand."""
        potential_winners = self.player_cycle.get_potential_winners()

        potential_winner_idx = [i for i, potential_winner in enumerate(potential_winners) if potential_winner]
        if sum(potential_winners) == 1:
            winner_ix = [i for i, active in enumerate(potential_winners) if active][0]
            winning_card_type = 'Only remaining player in round'

        else:
            assert self.stage == Stage.SHOWDOWN
            remaining_player_winner_ix, winning_card_type = get_winner([player.cards
                                                                        for ix, player in enumerate(self.players) if
                                                                        potential_winners[ix]],
                                                                       self.table_cards)
            winner_ix = potential_winner_idx[remaining_player_winner_ix]
        log.info(f"Player {winner_ix} won: {winning_card_type}")
        return winner_ix

    def _award_winner(self, winner_ix):
        """Hand the pot to the winner and handle side pots"""
        max_win_per_player_for_winner = self.player_max_win[winner_ix]
        total_winnings = sum(np.minimum(max_win_per_player_for_winner, self.player_max_win))
        remains = np.maximum(0, np.array(self.player_max_win) - max_win_per_player_for_winner)  # to be returned

        self.players[winner_ix].stack += total_winnings
        self.winner_ix = winner_ix
        if total_winnings < sum(self.player_max_win):
            log.info("Returning side pots")
            for i, player in enumerate(self.players):
                player.stack += remains[i]

    def _next_dealer(self):
        self.dealer_pos = self.player_cycle.next_dealer().seat

    def _next_player(self):
        """Move to the next player."""
        log.debug(f"Min call = {self.min_call}")
        log.debug(f"Player pots = {self.player_pots}")

        self.current_player = self.player_cycle.next_player(self.player_pots, self.min_call)

        if not self.current_player:
            if sum(self.player_cycle.alive) < 2:
                log.info("Only one player remaining in round")
                self.stage = Stage.END_HIDDEN
            else:
                log.info("End round - no current player returned")
                self._end_round()
                # todo: in some cases no new round should be initialized bc only one player is playing only it seems
                self._initiate_round()

        elif self.current_player == 'max_steps_total' or self.current_player == 'max_steps_after_raiser':
            log.debug(self.current_player)
            log.info("End of round ")
            self._end_round()
            return

    def _get_legal_moves(self):
        """Determine what moves are allowed in the current state."""
        self.legal_moves = []
        if self.player_pots[self.current_player.seat] == max(self.player_pots):
            self.legal_moves.append(Action.CHECK)
        else:
            self.legal_moves.append(Action.CALL)
            self.legal_moves.append(Action.FOLD)

        if self.current_player.stack >= 3 * self.big_blind - self.player_pots[self.current_player.seat]:
            if self.current_player.stack >= (self.community_pot + self.current_round_pot) >= self.min_call:
                self.legal_moves.append(Action.RAISE_POT)

            if self.current_player.stack > 0:
                self.legal_moves.append(Action.ALL_IN)

        log.debug(f"Community + current round pot: {self.community_pot + self.current_round_pot}")

    def _close_round(self):
        """Put player pots into community pots."""
        self.player_pots = [0] * len(self.players)
        self.played_in_round = 0

    def _create_card_deck(self):
        values = "23456789TJQKA"
        suites = "CDHS"
        self.deck = []  # contains cards in the deck
        _ = [self.deck.append(x + y) for x in values for y in suites]

    def _distribute_cards(self):
        log.info(f"Dealer is at position {self.dealer_pos}")
        for player in self.players:
            player.cards = []
            if player.stack <= 0:
                continue
            for _ in range(2):
                card = np.random.randint(0, len(self.deck))
                player.cards.append(self.deck.pop(card))
            log.info(f"Player {player.seat} got {player.cards} and ${player.stack}")

    def _distribute_cards_to_table(self, amount_of_cards):
        for _ in range(amount_of_cards):
            card = np.random.randint(0, len(self.deck))
            self.table_cards.append(self.deck.pop(card))
        log.info(f"Cards on table: {self.table_cards}")

    def render(self, mode='human'):
        """Render the current state."""
        screen_width = 600
        screen_height = 400
        table_radius = 200
        face_radius = 10

        if self.viewer is None:
            self.viewer = PygletWindow(screen_width + 50, screen_height + 50)
        self.viewer.reset()
        self.viewer.circle(screen_width / 2, screen_height / 2, table_radius, color=BLUE,
                           thickness=0)

        for i in range(len(self.players)):
            degrees = i * (360 / len(self.players))
            radian = (degrees * (np.pi / 180))
            x = (face_radius + table_radius) * np.cos(radian) + screen_width / 2
            y = (face_radius + table_radius) * np.sin(radian) + screen_height / 2
            if self.player_cycle.alive[i]:
                color = GREEN
            else:
                color = RED
            self.viewer.circle(x, y, face_radius, color=color, thickness=2)

            try:
                if i == self.current_player.seat:
                    self.viewer.rectangle(x - 60, y, 150, -50, (255, 0, 0, 10))
            except AttributeError:
                pass
            self.viewer.text(f"{self.players[i].name}", x - 60, y - 15,
                             font_size=10,
                             color=WHITE)
            self.viewer.text(f"Player {self.players[i].seat}: {self.players[i].cards}", x - 60, y,
                             font_size=10,
                             color=WHITE)
            equity_alive = int(round(float(self.players[i].equity_alive) * 100))

            self.viewer.text(f"${self.players[i].stack} (EQ: {equity_alive}%)", x - 60, y + 15, font_size=10,
                             color=WHITE)
            try:
                self.viewer.text(self.players[i].last_action_in_stage, x - 60, y + 30, font_size=10, color=WHITE)
            except IndexError:
                pass
            x_inner = (-face_radius + table_radius - 60) * np.cos(radian) + screen_width / 2
            y_inner = (-face_radius + table_radius - 60) * np.sin(radian) + screen_height / 2
            self.viewer.text(f"${self.player_pots[i]}", x_inner, y_inner, font_size=10, color=WHITE)
            self.viewer.text(f"{self.table_cards}", screen_width / 2 - 40, screen_height / 2, font_size=10,
                             color=WHITE)
            self.viewer.text(f"${self.community_pot}", screen_width / 2 - 15, screen_height / 2 + 30, font_size=10,
                             color=WHITE)
            self.viewer.text(f"${self.current_round_pot}", screen_width / 2 - 15, screen_height / 2 + 50,
                             font_size=10,
                             color=WHITE)

            x_button = (-face_radius + table_radius - 20) * np.cos(radian) + screen_width / 2
            y_button = (-face_radius + table_radius - 20) * np.sin(radian) + screen_height / 2
            try:
                if i == self.player_cycle.dealer_idx:
                    self.viewer.circle(x_button, y_button, 5, color=BLUE, thickness=2)
            except AttributeError:
                pass

        self.viewer.update()


class PlayerCycle:
    """Handle the circularity of the table."""

    def __init__(self, lst, start_idx=0, dealer_idx=0, max_steps_total=None,
                 last_raiser_step=None, max_steps_after_raiser=None, max_steps_after_big_blind=None):
        """Cycle over a list."""
        self.lst = lst
        self.start_idx = start_idx
        self.size = len(lst)
        self.max_steps_total = max_steps_total
        self.last_raiser_step = last_raiser_step
        self.max_steps_after_raiser = max_steps_after_raiser
        self.max_steps_after_big_blind = max_steps_after_big_blind
        self.last_raiser = None
        self.step_counter = 0
        self.steps_for_blind_betting = 2
        self.second_round = False
        self.idx = 0
        self.dealer_idx = dealer_idx
        self.can_still_make_moves_in_this_hand = []  # if the player can still play in this round
        self.alive = [True] * len(self.lst)  # if the player can still play in the following rounds
        self.out_of_cash_but_contributed = [False] * len(self.lst)
        self.new_hand_reset()
        self.checkers = 0
        self.folder = None

    def new_hand_reset(self):
        """Reset state if a new hand is dealt."""
        self.idx = self.start_idx
        self.can_still_make_moves_in_this_hand = [True] * len(self.lst)
        self.out_of_cash_but_contributed = [False] * len(self.lst)
        self.folder = [False] * len(self.lst)
        self.step_counter = 0

    def new_round_reset(self):
        """Reset the state for the next stage: flop, turn or river."""
        self.step_counter = 0
        self.second_round = False
        self.idx = self.dealer_idx
        self.last_raiser_step = len(self.lst)
        self.checkers = 0

    def next_player(self, player_pots: List[float], min_call: float, step: int = 1):
        """Switch to the next player in the round."""
        if sum(np.array(self.can_still_make_moves_in_this_hand) + np.array(self.out_of_cash_but_contributed)) < 2:
            log.debug("Only one player remaining")
            return False

        log.debug("==== Next player logic")
        log.debug(f"Start index = {self.idx}")
        self.idx += step
        self.step_counter += step
        self.idx %= len(self.lst)
        log.debug(f"End index = {self.idx}")

        if self.step_counter > len(self.lst):
            self.second_round = True

        has_players_with_more_turns = False
        orig_idx = self.idx

        num_active_players = (
            sum(np.logical_and(
                np.array(self.can_still_make_moves_in_this_hand), np.logical_not(np.array(self.out_of_cash_but_contributed)))
            )
        )
        if num_active_players == 1:
            log.debug("Only one active player")

        while True:
            if self.can_still_make_moves_in_this_hand[self.idx] and self.lst[self.idx].stack > 0:
                # this should only include players who haven't folded with non-zero stacks
                log.debug(f"Player stack: {self.lst[self.idx].stack}")
                log.debug(f"Action = {self.lst[self.idx].last_action_in_stage}")
                if not self.lst[self.idx].last_action_in_stage and num_active_players > 1:
                    log.debug("First move")
                    has_players_with_more_turns = True
                    break
                if player_pots[self.idx] < min_call:
                    log.debug(f"Player {self.idx} is not caught up on the action")
                    has_players_with_more_turns = True
                    break
                if player_pots[self.idx] == min_call:
                    if self.lst[self.idx].last_action_in_stage in [ActionBlind.SMALL.name, ActionBlind.BIG.name]:
                        log.debug(f"Player {self.idx} has matched funds, but they haven't acted yet in the round")
                        has_players_with_more_turns = True
                        break

            self.idx += 1
            self.step_counter += 1
            self.idx %= len(self.lst)

            if self.idx == orig_idx:
                log.debug("Ran through all the players")
                break

        if not has_players_with_more_turns:
            return False

        self.update_alive()

        return self.lst[self.idx]

    def next_dealer(self):
        """Move the dealer to the next player that's still in the round."""
        self.dealer_idx += 1
        self.dealer_idx %= len(self.lst)

        while True:
            if self.can_still_make_moves_in_this_hand[self.dealer_idx]:
                break

            self.dealer_idx += 1
            self.dealer_idx %= len(self.lst)

        return self.lst[self.dealer_idx]

    def set_idx(self, idx):
        """Set the index to a specific player."""
        self.idx = idx

    def deactivate_player(self, idx):
        """Deactivate a pleyr if he has folded or is out of cash."""
        assert self.can_still_make_moves_in_this_hand[idx], "Already deactivated"
        self.can_still_make_moves_in_this_hand[idx] = False

    def deactivate_current(self):
        """Deactivate the current player if he has folded or is out of cash."""
        assert self.can_still_make_moves_in_this_hand[self.idx], "Already deactivated"
        self.can_still_make_moves_in_this_hand[self.idx] = False

    def mark_folder(self):
        """Mark a player as no longer eligible to win cash from the current hand."""
        self.folder[self.idx] = True

    def mark_raiser(self):
        """Mark a raise for the current player."""
        if self.step_counter > 2:
            self.last_raiser = self.step_counter

    def mark_checker(self):
        """Counter the number of checks in the round."""
        self.checkers += 1

    def mark_out_of_cash_but_contributed(self):
        """Mark current player as a raiser or caller, but is out of cash."""
        self.out_of_cash_but_contributed[self.idx] = True
        self.deactivate_current()

    def mark_bb(self):
        """Ensure bb can raise."""
        self.last_raiser_step = self.step_counter + len(self.lst)
        self.max_steps_total = self.step_counter + len(self.lst) * 2

    def is_raising_allowed(self):
        """Check if raising is still allowed at this position."""
        return self.step_counter <= self.last_raiser_step

    def update_alive(self):
        """Update the alive property."""
        self.alive = np.array(self.can_still_make_moves_in_this_hand) + \
                     np.array(self.out_of_cash_but_contributed)

    def get_potential_winners(self):
        """Players eligible to win the pot."""
        potential_winners = np.logical_and(np.logical_or(np.array(self.can_still_make_moves_in_this_hand),
                                                         np.array(self.out_of_cash_but_contributed)),
                                           np.logical_not(np.array(self.folder)))
        return potential_winners


class PlayerShell:
    """Player shell."""

    def __init__(self, stack_size, name):
        """Initiaization of an agent."""
        self.stack = stack_size
        self.seat = None
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.agent_obj = None
