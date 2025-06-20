import numpy as np

from typing import Protocol
from .layers import Case1Layer, Case2Layer, Case3Layer, TubeLayer
from classical_laminate_theory import LaminateLayer

class Layer(Protocol):
    thickness: float
    C11: float
    C22: float
    C33: float
    C12: float
    C13: float
    C23: float
    thick_thermal_expansion_vector: np.ndarray
    C_bar: np.ndarray


class LayerFactory:
    @classmethod
    def create_layer(self, layer: Layer) -> TubeLayer:
        return self.get_appropriate_class(layer)(
            *self.extract_layer_properties(layer)
        )

    @staticmethod
    def get_appropriate_class(layer: Layer) -> TubeLayer:
        if layer.C22 == layer.C33:
            return Case3Layer
        if layer.C22 / layer.C33 >= 0:
            return Case1Layer
        return Case2Layer

    @classmethod
    def extract_layer_properties(cls, layer: Layer):
        return [
            layer.thickness,
            layer.C11,
            layer.C22,
            layer.C33,
            layer.C12,
            layer.C13,
            layer.C23,
            cls.compute_xi(layer),
        ]

    @staticmethod
    def compute_xi(layer: Layer):
        return np.dot(layer.C_bar, layer.thick_thermal_expansion_vector)


def main():
    pass


if __name__ == "__main__":
    main()


# End
