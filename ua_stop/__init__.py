"""UA-SEA-RAFT: uncertainty-aware adaptive iteration stopping for SEA-RAFT.

The package is split in two halves on purpose:

* GPU half   -- ``model_wrapper``, ``data``, ``trace``, ``latency``,
  ``tile_stop``, ``uncertainty``, ``hooks``. These need torch and the upstream
  submodule.
* CPU half   -- ``criterion``, ``sweep``, ``conformal``, ``plots``, ``utils``.
  Pure numpy/scipy/matplotlib, so every threshold can be replayed offline from
  a cached trace without touching a GPU.

That split is what makes the method cheap to study: one full-budget GPU pass
produces the trace, and thousands of stopping configurations are then evaluated
on a laptop.
"""

__version__ = "0.1.0"

from .criterion import (  # noqa: F401
    StopConfig,
    decide_stop,
    decide_stop_batch,
    iters_of,
)

__all__ = [
    "__version__",
    "StopConfig",
    "decide_stop",
    "decide_stop_batch",
    "iters_of",
]
