from dataclasses import dataclass


@dataclass(order=True)
class ObservationModel:
    environment_name: str = ''
    encoding_group: str = 0
