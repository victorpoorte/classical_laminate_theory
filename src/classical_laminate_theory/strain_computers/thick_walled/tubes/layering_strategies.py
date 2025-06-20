from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Protocol

from classical_laminate_theory import create_complete_layers


class Lamina(Protocol):
    ...


class Layer(Protocol):
    lamina: Lamina
    thickness: float
    rotation: float
    degrees: bool


class VesselLayeringStrategy(Protocol):
    layering_pattern: list[float]

    @property
    def layers_per_winding_angle(self) -> int:
        return len(self.layering_pattern)

    @abstractmethod
    def create_layers(self, layers: list[Layer]) -> list[Layer]:
        ...

@dataclass
class LaminatedStrategy(VesselLayeringStrategy):
    layering_pattern: list[float] = field(default_factory=lambda: [1])

    def create_layers(layers: list[Layer]) -> list[Layer]:
        return layers


@dataclass
class FilamentWoundStrategy(VesselLayeringStrategy):
    layering_pattern: list[float] = field(
        default_factory=lambda: [1, -1, -1, 1]
    )

    def create_layers(self, layers: list[Layer]) -> list[Layer]:
        winding_angles = [
            sign * layer.rotation
            for layer in layers
            for sign in self.layering_pattern
        ]
        thicknesses = [
            layer.thickness / self.layers_per_winding_angle
            for layer in layers
            for _ in self.layering_pattern
        ]
        materials = [
            layer.lamina for layer in layers for _ in self.layering_pattern
        ]
        degrees = [
            layer.degrees for layer in layers for _ in self.layering_pattern
        ]
        return create_complete_layers(
            winding_angles, thicknesses, materials, degrees
        )


def main():
    pass


if __name__ == "__main__":
    main()


# End
