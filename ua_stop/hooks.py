"""Fallback stopping signal: the recurrent update itself.

If every MoL read-out turns out to be degenerate on a given checkpoint (see the
long note at the top of ``uncertainty.py``), the method still has a signal that
needs no uncertainty head at all: the norm of the hidden-state update of the
recurrent block. It is captured with a forward hook, so no upstream file is
touched and the numerics of the forward pass are untouched as well.

This is the project's insurance policy. It is reported in the paper as an
alternative signal, not as the headline, because it carries no calibrated
notion of "how wrong am I" -- only "how much am I still moving".

Usage
-----
    probe = HiddenStateProbe().attach(wrapper.model)
    out = wrapper.forward_trace(img1, img2)
    series = probe.series()      # RMS ||h_i - h_{i-1}|| per iteration
    probe.detach()
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch

#: Substrings matched (case-insensitively) against ``named_modules()``.
DEFAULT_PATTERNS: Tuple[str, ...] = ("update", "gru", "convnext", "refine")


class HiddenStateProbe:
    """Records the per-call change of a module's output tensor."""

    def __init__(self, patterns: Sequence[str] = DEFAULT_PATTERNS) -> None:
        self.patterns = tuple(str(p).lower() for p in patterns)
        self.module_name: Optional[str] = None
        self._handles: List[Any] = []
        self._prev: Optional[torch.Tensor] = None
        self._norms: List[float] = []

    # -- wiring ---------------------------------------------------------

    def _pick(self, model: torch.nn.Module):
        matches = []
        for name, module in model.named_modules():
            if not name:
                continue
            low = name.lower()
            if any(p in low for p in self.patterns):
                matches.append((len(name), name, module))
        if not matches:
            return None, None
        matches.sort(key=lambda t: (t[0], t[1]))  # shallowest match wins
        return matches[0][1], matches[0][2]

    def attach(self, model: torch.nn.Module) -> "HiddenStateProbe":
        name, module = self._pick(model)
        if module is None:
            raise RuntimeError(
                "no submodule name matched %s; inspect model.named_modules() "
                "and pass patterns=(...)" % (self.patterns,)
            )
        self.module_name = name
        self._handles.append(module.register_forward_hook(self._hook))
        self.reset()
        return self

    def detach(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []

    def reset(self) -> None:
        self._prev = None
        self._norms = []

    # -- capture --------------------------------------------------------

    @staticmethod
    def _tensor_of(output: Any) -> Optional[torch.Tensor]:
        if torch.is_tensor(output):
            return output
        if isinstance(output, dict):
            for value in output.values():
                if torch.is_tensor(value):
                    return value
        if isinstance(output, (list, tuple)):
            for value in output:
                if torch.is_tensor(value):
                    return value
        return None

    def _hook(self, module: torch.nn.Module, inputs: Any, output: Any) -> None:
        tensor = self._tensor_of(output)
        if tensor is None:
            return
        current = tensor.detach().float()
        if self._prev is not None and self._prev.shape == current.shape:
            diff = (current - self._prev).norm(p=2)
            rms = float(diff.item()) / max(1.0, float(current.numel()) ** 0.5)
            self._norms.append(rms)
        else:
            self._norms.append(float("inf"))  # first call has no predecessor
        self._prev = current

    # -- read-out -------------------------------------------------------

    @property
    def norms(self) -> List[float]:
        return list(self._norms)

    def series(self) -> np.ndarray:
        """(n_calls,) RMS hidden-state update; entry 0 is ``inf`` by convention,
        exactly like ``D`` in :mod:`ua_stop.criterion`, so the same stopping
        rule can consume it unchanged."""
        return np.asarray(self._norms, dtype=np.float64)

    def __enter__(self) -> "HiddenStateProbe":
        return self

    def __exit__(self, *exc) -> bool:
        self.detach()
        return False
