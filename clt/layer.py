from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .laminate_layer import LaminateLayer
from .material import Lamina


@dataclass
class Layer:
    lamina: Lamina
    thickness: float
    rotation: float
    degrees: bool

def layer_to_laminate_layer(layer: Layer) -> LaminateLayer:
    return LaminateLayer(**layer.__dict__)


def create_complete_layers(
    rotations: list[float],
    thicknesses: float | list[float],
    materials: Lamina | list[Lamina],
    degrees: bool | list[bool],
) -> list[Layer]:
    number_of_orientations = len(rotations)
    thicknesses = _check_list(thicknesses, number_of_orientations)
    materials = _check_list(materials, number_of_orientations)
    degrees = _check_list(degrees, number_of_orientations)
    return [
        Layer(material, thickness, orientation, degree)
        for orientation, thickness, material, degree in zip(
            rotations, thicknesses, materials, degrees
        )
    ]


def _check_list(
    my_list: list | Lamina | float | bool, number_of_orientations: int
) -> list[float] | list[Lamina] | list[bool]:
    if isinstance(my_list, list) or isinstance(my_list, np.ndarray):
        if len(my_list) != number_of_orientations:
            raise ValueError("Length of lists do not match...")
        return my_list
    return [my_list for _ in range(number_of_orientations)]


def main():
    pass


if __name__ == "__main__":
    main()


# End
