from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

LOAD_ENTRIES = ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy", "operating_temperature", "reference_temperature"]


@dataclass
class LaminateLoad:
    Nx: float = 0
    Ny: float = 0
    Nxy: float = 0
    Mx: float = 0
    My: float = 0
    Mxy: float = 0
    operating_temperature: float = 0
    reference_temperature: float = 0

    def __lt__(self, other: LaminateLoad) -> bool:
        return self.magnitude < other.magnitude
    
    @property
    def magnitude(self):
        return np.sqrt(
            self.Nx**2 + self.Ny**2 + self.Nxy**2 + self.Mx**2 + self.My**2 + self.Mxy**2
        )

    @property
    def temperature_delta(self):
        return self.operating_temperature - self.reference_temperature

    @cached_property
    def vector(self):
        return np.array(
            [
                [self.Nx],
                [self.Ny],
                [self.Nxy],
                [self.Mx],
                [self.My],
                [self.Mxy],
            ]
        )

    def scale(self, factor: float) -> LaminateLoad:
        entries = {
            entry: getattr(self, entry) * factor for entry in LOAD_ENTRIES
        }
        return LaminateLoad(**entries)
    
    @classmethod
    def _create_angle_based_load(cls, x_axis: str, y_axis: str, angle: float) -> LaminateLoad:
        load_dict = {entry: 0 for entry in LOAD_ENTRIES}
        
        # Update load
        load_dict[x_axis] = np.cos(angle)
        load_dict[y_axis] = np.sin(angle)
        load = cls(**load_dict)

        return load


def convert_load_array_to_laminate_load(load: np.ndarray | LaminateLoad) -> LaminateLoad:
    if isinstance(load, LaminateLoad): return load
    load = LaminateLoad(*load)
    return load


def main():
    pass


if __name__ == "__main__":
    main()


# End
