from __future__ import annotations
from dataclasses import dataclass

from typing import Protocol
from .layer import Layer


INTER_LAMINAR_THICKNESS_FACTOR = 1


class Lamina(Protocol):
    def create_inter_lamina_material(self) -> Lamina:
        ...

class LayeringStrategy(Protocol):
     def create_complete_layers(self, rotation: float | list[float], layer_thickness: float, material: Lamina, degrees: bool):
        ...

class LaminateLayerStrategy(LayeringStrategy):
    def create_layers(self, layers: list[Layer]) -> list[Layer]:
        return layers
    
    def create_complete_layers(self, angles, layer_thickness, material, degrees) -> list[Layer]:
        return [
            Layer(material, layer_thickness, angle, degrees)
            for angle in angles
        ]


@dataclass
class LaminateInterLaminarStrategy(LaminateLayerStrategy):
    inter_laminar_thickness_factor: float = INTER_LAMINAR_THICKNESS_FACTOR

    def create_layers(self, layers: list[Layer]) -> list[Layer]:
        return super().create_layers(
            self._create_layers_with_inter_lamina_layers(layers)
        )

    def _create_layers_with_inter_lamina_layers(self, layers):
        orientations = self._create_orientations_list(layers)
        thicknesses = self._create_thicknesses_list(layers)
        laminae = self._create_materials_list(layers)
        degrees = self._create_degrees_list(layers)
        return [
            Layer(lamina, thickness, orientation, degree)
            for orientation, thickness, lamina, degree in zip(
                orientations, thicknesses, laminae, degrees
            )
        ]

    def _create_degrees_list(self, layers: list[Layer]):
        degrees = [layer.degrees for layer in layers for _ in range(2)]
        degrees.pop()
        return degrees

    def _create_materials_list(self, layers: list[Layer]):
        materials = [
            lamina
            for layer in layers
            for lamina in (
                layer.lamina,
                layer.lamina.create_inter_lamina_material(),
            )
        ]
        materials.pop()
        return materials

    def _create_thicknesses_list(self, layers: list[Layer]):
        thicknesses = [
            thickness
            for layer in layers
            for thickness in (
                layer.thickness,
                layer.thickness * self.inter_laminar_thickness_factor,
            )
        ]
        thicknesses.pop()
        return thicknesses

    def _create_orientations_list(self, layers: list[Layer]) -> list[float]:
        orientations = [
            orientation
            for layer in layers
            for orientation in (layer.rotation, 0)
        ]
        orientations.pop()
        return orientations
    

class LayeringStrategyFactory():
    _strategies = {
        "Laminated": LaminateLayerStrategy(),
    }

    @property
    def available(self):
        return " ,".join(self._strategies.keys())

    def get_layering_strategy(self, strategy: str) -> LayeringStrategy:
        strat = self._strategies.get(strategy)
        if strat is None:
            raise ValueError(
                f"\n{strategy} is an invalid layering strategy...\n"
                + f"Available strategies are: {self.available}"
            )
        return strat


def main():
    pass


if __name__ == "__main__":
    main()


# End
