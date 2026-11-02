""" Streaming data access interface. """

from collections.abc import Iterator
import numpy as np

from .types import BootstrapWindow, EvalTarget, StreamWindow

class DataStream:
    """ Provide experiment windows separately from their evaluation labels. """
    def __init__(
        self,
        bootstrap_window: BootstrapWindow,
        stream_windows: list[StreamWindow],
        eval_target: dict[int, EvalTarget],
    ) -> None:
        self._bootstrap_window = bootstrap_window
        self._stream_windows = tuple(stream_windows)
        self._eval_target = dict(eval_target)

        self._validate()

    def bootstrap(self) -> BootstrapWindow:
        """ Return the labeled W0 window. """
        return self._bootstrap_window

    def __iter__(self) -> Iterator[StreamWindow]:
        """ Return feature-only W1, W2, ... windows. """
        return iter(self._stream_windows)

    def __len__(self) -> int:
        """ Return the number of online windows excluding W0. """
        return len(self._stream_windows)

    def eval_target(self, window_id: int) -> EvalTarget:
        """ Return evaluation labels for the specified window. """
        return self._eval_target[window_id]

    def _validate(self) -> None:
        """ Check that online windows and evaluation labels correspond exactly. """
        for window in self._stream_windows:
            if window.window_id == 0:
                raise ValueError("W0 must not appear among online windows")

            target = self.eval_target(window.window_id)

            # Check window row IDs against evaluation targets.
            if not np.array_equal(window.row_ids, target.row_ids):
                raise ValueError("Online windows do not match evaluation targets")
