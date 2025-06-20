from typing import Protocol
from .layers import TubeLayer


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


class LayerFactory:
    @classmethod
    def create_layer(cls, layer: Layer) -> TubeLayer:
        return TubeLayer(*cls.extract_layer_properties(layer))

    @staticmethod
    def extract_layer_properties(layer: Layer) -> list[float]:
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
        ]


def main():
    pass


if __name__ == "__main__":
    main()


# End
