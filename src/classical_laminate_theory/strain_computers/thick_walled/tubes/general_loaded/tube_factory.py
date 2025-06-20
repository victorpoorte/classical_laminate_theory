from typing import Protocol
from ..tube_factory import VesselFactory
from .layers_factory import LayerFactory
from .tube import Tube


class Layer(Protocol):
    ...


class VesselFactory(VesselFactory):
    def __init__(self) -> None:
        super().__init__(Tube, LayerFactory())

    def determine_number_of_equations(self, number_of_layer: int) -> int:
        return number_of_layer * 2 + 2


def main():
    pass


if __name__ == "__main__":
    main()

# End
