from typing import Protocol

from .tube import Tube


class Layer(Protocol):
    ...


class TubeLayer(Protocol):
    ...


class VesselLayerFactory(Protocol):
    def create_layer(self, layer: Layer) -> TubeLayer:
        ...


class VesselFactory:
    def __init__(
        self,
        tube_initialiser: Tube,
        vessel_layer_factory: VesselLayerFactory,
    ) -> None:
        self.tube_initialiser = tube_initialiser
        self.layer_factory = vessel_layer_factory

    def create_vessel(
        self, inner_radius: float, layers: list[Layer]
    ) -> Tube:
        vessel_layers = [
            self.layer_factory.create_layer(layer)
            for layer in layers
        ]
        return self.tube_initialiser(inner_radius, vessel_layers)

    def determine_number_of_equations(self, number_of_layers: int) -> int:
        return self.tube_initialiser.determine_number_of_equations(
            number_of_layers
        )


def main():
    pass


if __name__ == "__main__":
    main()


# End
