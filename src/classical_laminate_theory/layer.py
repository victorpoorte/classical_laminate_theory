from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .laminate_layer import LaminateLayer
from .material import Lamina
from .laminate import LaminateStack



# class LaminateStack(Protocol):
#     angles: list[float]
#     layer_thicknesses: list[float]
#     material_names: list[str]
#     degrees: bool


class MaterialInitialiser(Protocol):
    def get(self, name: str) -> Lamina: ...


class LayeringStrategy(Protocol):
    def create_complete_layers(self, angles: list[float], layer_thicknesses: list[float], materials: list[Lamina], degrees: bool) -> list[Layer]: ...


class LayeringStrategyFactory(Protocol):
    def create(self, name: str) -> LayeringStrategy: ...


@dataclass
class Layer:
    lamina: Lamina
    thickness: float
    rotation: float
    degrees: bool


def layer_to_laminate_layer(layer: Layer) -> LaminateLayer:
    return LaminateLayer(**layer.__dict__)


@dataclass
class LayersBuilder:
    laminate_stacks: list[LaminateStack]
    material: MaterialInitialiser
    strategy_factory: LayeringStrategyFactory

    @property
    def _strategies(self):
        return [
            self.strategy_factory.create(lam.layering_strategy)
            for lam in self.laminate_stacks
        ]

    def build(self):
        return [
            strategy.create_complete_layers(
                laminate.angles,
                laminate.layer_thicknesses,
                [self.material.get(mat) for mat in laminate.material_names],
                laminate.degrees 
            )
            for strategy, laminate in zip(self._strategies, self.laminate_stacks)
        ]


def main():
    pass


if __name__ == "__main__":
    main()


# End
