from typing import Protocol
from .tube_factory import VesselFactory

from .general_loaded.tube import Tube
from .general_loaded.layers_factory import LayerFactory
from .layering_strategies import LaminatedStrategy, FilamentWoundStrategy

# from ..clt import TraditionLaminateFactory, LaminateWithInterLayersFactory
# Need to think whether it makes sense to keep this file...


class Layer(Protocol):
    ...


class LaminatedVesselFactory(VesselFactory):
    def __init__(self) -> None:
        tube_initialiser = Tube
        vessel_layer_factory = LayerFactory()
        layering_strategy = LaminatedStrategy(TraditionLaminateFactory())
        super().__init__(
            tube_initialiser, vessel_layer_factory, layering_strategy
        )

    def determine_number_of_equations(self, number_of_layers: int) -> int:
        return self.tube_initialiser.determine_number_of_equations(
            number_of_layers
        )


class FilamentWoundVesselFactory(VesselFactory):
    def __init__(self) -> None:
        tube_initialiser = Tube
        vessel_layer_factory = LayerFactory()
        layering_strategy = FilamentWoundStrategy(TraditionLaminateFactory())
        super().__init__(
            tube_initialiser, vessel_layer_factory, layering_strategy
        )


class LaminatedVesselFactoryWithIntraLaminaLayers(VesselFactory):
    def __init__(self) -> None:
        tube_initialiser = Tube
        vessel_layer_factory = LayerFactory()
        layering_strategy = LaminatedStrategy(LaminateWithInterLayersFactory())
        super().__init__(
            tube_initialiser, vessel_layer_factory, layering_strategy
        )


class FilamentWoundVesselFactoryWithIntraLaminarLayers(VesselFactory):
    def __init__(self) -> None:
        raise NotImplementedError("This still needs to be implemented...")
        tube_initialiser = Tube
        vessel_layer_factory = LayerFactory()
        layering_strategy = FilamentWoundStrategy(
            LaminateWithInterLayersFactory()
        )
        super().__init__(
            tube_initialiser, vessel_layer_factory, layering_strategy
        )


def main():
    pass


if __name__ == "__main__":
    main()


# End
