"""Processors to process actions, observations, etc., each call."""

import numpy as np
from rl.core import Processor

from gym_env.action import Action
from gym_env.observation import Observation


class LegalMovesProcessor(Processor):
    """Processor to get legal moves."""

    def __init__(self, num_opponents: int, num_actions: int):
        """initialize properties."""
        observation = Observation(num_opponents, num_actions)
        self.legal_move_indices = np.array([observation.labels().index(f"is_legal_action_{i + 1}") for i in range(num_actions)])
        self.legal_moves = None

    def process_state_batch(self, batch):
        """Remove second dimension to make it possible to pass it into CNN."""
        return np.squeeze(batch, axis=1)

    def process_observation(self, observation):
        """Process the observation."""
        legal_moves_idx = np.nonzero(observation[self.legal_move_indices])[0]
        self.legal_moves = [Action(idx) for idx in legal_moves_idx]
        return observation

    def process_info(self, info):
        # just patch this for now
        _ = info
        return {'x': 1}

    def process_action(self, action):
        """Find nearest legal action."""
        if self.legal_moves is None:
            return action

        action = Action(action)
        if action in self.legal_moves:
            return action.value

        if action == Action.FOLD:
            return Action.CHECK.value
        elif action == Action.CHECK:
            return Action.FOLD.value
        elif action == Action.CALL:
            return Action.CHECK.value
        elif action == Action.RAISE_POT:
            if Action.CALL in self.legal_moves:
                return Action.CALL.value
            elif Action.CHECK in self.legal_moves:
                return Action.CHECK.value
            elif Action.FOLD in self.legal_moves:
                return Action.FOLD.value
            raise RuntimeError(f"Could not process the action {action}. Legal moves = {self.legal_moves}")
        elif action == Action.ALL_IN:
            if Action.RAISE_POT in self.legal_moves:
                return Action.RAISE_POT.value
            elif Action.CALL in self.legal_moves:
                return Action.CALL.value
            elif Action.CHECK in self.legal_moves:
                return Action.CHECK.value
            elif Action.FOLD in self.legal_moves:
                return Action.FOLD.value
            raise RuntimeError(f"Could not process the action {action}. Legal moves = {self.legal_moves}")

        raise RuntimeError(f"Could not process the action {action}. Legal moves = {self.legal_moves}")


class SingleOpponentLegalMovesProcessor(LegalMovesProcessor):
    """Processor to get legal moves and reduce space down to one effective opponent."""

    def __init__(self, num_opponents: int, num_actions: int):
        """initialize properties."""
        super().__init__(num_opponents, num_actions)

        observation = Observation(num_opponents, num_actions)

        # consolidate stacks and blinds
        self.num_obs_to_keep = observation.ndim() - 2 * (num_opponents - 1)
        self.idx_stack_1 = observation.labels().index('opp_stack_1')
        self.idx_blind_1 = observation.labels().index('opp_blind_1')
        self.idx_other_stacks = []
        self.idx_other_blinds = []
        for i in range(1, num_opponents):
            self.idx_other_stacks.append(observation.labels().index(f'opp_stack_{i + 1}'))
            self.idx_other_blinds.append(observation.labels().index(f'opp_blind_{i + 1}'))

    def process_observation(self, observation):
        """Process the observation."""
        _ = super().process_observation(observation)

        # add opponent stacks and blinds to opponent 1
        observation[self.idx_stack_1] += sum(observation[self.idx_other_stacks])
        observation[self.idx_blind_1] += sum(observation[self.idx_other_blinds])

        return observation[:self.num_obs_to_keep]
