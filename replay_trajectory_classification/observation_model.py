from dataclasses import dataclass


@dataclass(order=True)
class ObservationModel:
    """Determines which environment and data points data correspond to."""

    environment_name: str = ""
    encoding_group: str = 0
