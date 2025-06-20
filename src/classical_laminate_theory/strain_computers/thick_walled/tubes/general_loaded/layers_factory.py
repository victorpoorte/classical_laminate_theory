from typing import Protocol

import numpy as np

from classical_laminate_theory import layer_to_laminate_layer


from .layers import Case1Layer, Case2Layer, Case3Layer, TubeLayer


class Layer(Protocol):
    thickness: float
    C11: float
    C22: float
    C33: float
    C12: float
    C13: float
    C23: float
    C16: float
    C26: float
    C36: float
    C66: float
    thick_thermal_expansion_vector: np.ndarray
    C_bar: np.ndarray


class LayerFactory:
    @classmethod
    def create_layer(cls, layer: Layer) -> TubeLayer:
        layer = layer_to_laminate_layer(layer)
        initialiser = cls.get_layer_initialiser(layer)
        args = cls.extract_layer_properties(layer)
        return initialiser(*args)

    @staticmethod
    def get_layer_initialiser(layer: Layer) -> TubeLayer:
        ratio = layer.C22 / layer.C33
        if ratio == 1:
            return Case2Layer
        if ratio > 0:
            return Case1Layer
        if ratio < 0:
            return Case3Layer
        
        raise ValueError("Invalid beta value")

    @classmethod
    def extract_layer_properties(cls, layer: Layer) -> list[float]:
        return [
            layer.thickness,
            layer.C11,
            layer.C22,
            layer.C33,
            layer.C12,
            layer.C13,
            layer.C23,
            layer.C16,
            layer.C26,
            layer.C36,
            layer.C66,
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
