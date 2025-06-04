import builtins
from unittest.mock import patch

import pytest

from day1_0_guessing_game import get_user_guess


def test_invalid_then_valid():
    with patch('builtins.input', side_effect=['abc', '42']):
        assert get_user_guess(1, 100) == 42


def test_multiple_retries():
    # first input is non-integer, second is out of range, third is valid
    with patch('builtins.input', side_effect=['xyz', '150', '75']):
        assert get_user_guess(1, 100) == 75
