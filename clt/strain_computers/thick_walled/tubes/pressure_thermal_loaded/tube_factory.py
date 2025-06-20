from ..tube_factory import VesselFactory
from .layers_factory import LayerFactory
from .tube import Tube


class TubeFactory(VesselFactory):
    def __init__(self) -> None:
        super().__init__(Tube, LayerFactory())


def main():
    pass


if __name__ == "__main__":
    main()

# End
