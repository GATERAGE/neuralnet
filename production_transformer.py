# SPDX-License-Identifier: Apache-2.0
# (c) 2024-2026 GATERAGE — neuralnet
#
# DEPRECATION SHIM (added in v0.1.0a3, removal scheduled for 0.2.0)
#
# The canonical content now lives at `neuralnet.transformer`.
# This shim re-exports the public symbols so existing imports keep working
# for one alpha cycle. Migrate to the package import to silence the
# DeprecationWarning:
#
#     from neuralnet.transformer import ...
#
"""Deprecation shim for `production_transformer`. See module docstring."""
from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "Importing `production_transformer` is deprecated since v0.1.0a3. "
    "Use `from neuralnet.transformer import ...` instead. "
    "This shim will be removed in 0.2.0.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralnet.transformer import *  # noqa: F401, F403
