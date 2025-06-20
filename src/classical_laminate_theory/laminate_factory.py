from typing import Protocol

from .layer import layer_to_laminate_layer
from .laminate import Laminate


from .layering_strategies import (
    LaminateLayerStrategy,
    LaminateInterLaminarStrategy,
)


class Layer(Protocol):
    ...


class LaminateFactoryProtocol(Protocol):
    laminate_layer_strategy: LaminateLayerStrategy

    def create_laminate(self, layers: list[Layer]) -> Laminate:
        layers = self.laminate_layer_strategy.create_layers(layers)
        laminate_layers = [
            layer_to_laminate_layer(layer)
            for layer in layers
        ]
        return Laminate(laminate_layers)


class TraditionalLaminateFactory(LaminateFactoryProtocol):
    laminate_layer_strategy = LaminateLayerStrategy()


class LaminateWithInterLayersFactory(LaminateFactoryProtocol):
    laminate_layer_strategy = LaminateInterLaminarStrategy()


def main():
    pass


if __name__ == "__main__":
    main()

# End
