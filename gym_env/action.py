"""Define the action space."""

from enum import Enum


class Action(Enum):
    """Allowed actions."""

    # use simplified set
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE_HALF_POT = 3
    RAISE_POT = 4
    ALL_IN = 5


class ActionBlind(Enum):
    """Action corresponding to blinds."""

    SMALL = 0
    BIG = 1
