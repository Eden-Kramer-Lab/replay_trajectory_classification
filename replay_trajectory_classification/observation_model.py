"""Class for representing observation models.

This module provides the ObservationModel class that pairs environments
with experimental conditions (trial types, etc.) for neural decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable


@dataclass(order=True)
class ObservationModel:
    """Determines which environment and data points data correspond to.

    Attributes
    ----------
    environment_name : str, optional
    encoding_group : Hashable, optional

    """

    environment_name: str = ""
    encoding_group: Hashable = 0
