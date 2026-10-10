"""Coverage tests for plumerank.segment.

Focuses on the defensive validation layer in _new_subtractor that would only be
reached if invalid arguments somehow bypassed the Segmenter.__init__ gate.
"""

from __future__ import annotations

import pytest

from plumerank.segment import METHODS, _new_subtractor


def test_new_subtractor_raises_on_unknown_method() -> None:
    """The _new_subtractor function validates its method argument independently.

    This is defensive programming: Segmenter.__init__ already validates, but if
    someone calls _new_subtractor directly with an invalid method, it raises
    immediately rather than silently using the wrong subtractor (which would
    publish a benchmarked row under the wrong label).

    This test reaches line 244: the fallback raise at the end of _new_subtractor.
    """
    with pytest.raises(ValueError, match="unknown method 'knn'"):
        _new_subtractor("knn")


def test_new_subtractor_raises_on_empty_method() -> None:
    """Empty string is not a valid method."""
    with pytest.raises(ValueError, match="unknown method ''"):
        _new_subtractor("")


def test_new_subtractor_raises_on_typo_in_valid_method() -> None:
    """A typo in a method name must raise, not silently fall back."""
    with pytest.raises(ValueError, match="unknown method"):
        _new_subtractor("mog")  # typo: should be "mog2"


def test_new_subtractor_error_mentions_valid_methods() -> None:
    """The error message should list the valid options so the caller knows what's accepted."""
    with pytest.raises(ValueError, match="expected one of"):
        _new_subtractor("invalid")


def test_new_subtractor_constructs_all_valid_methods() -> None:
    """Each of METHODS constructs successfully and returns a cv2.BackgroundSubtractor.

    This is not about line 244 (all these take the early returns), but it ensures
    that the three paths leading away from line 244 all work correctly.
    """
    for method in METHODS:
        subtractor = _new_subtractor(method)
        # Each should be a BackgroundSubtractor (duck-typed: has an apply method).
        assert hasattr(subtractor, "apply"), f"{method} did not construct a subtractor"
        assert callable(subtractor.apply), f"{method} subtractor.apply is not callable"
