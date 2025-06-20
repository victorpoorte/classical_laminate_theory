from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from classical_laminate_theory import create_complete_layers
from ..non_linear_solution_methods.non_linear_solution_methods.convergence_checkers import ForceResidualConvergence
from ..non_linear_solution_methods.non_linear_solution_methods.newton_raphson_with_residuals import NewtonRaphsonWithResiduals
from .thick_tube_strain_computer_protocol import ThickWalledStrainComputer
from ..tubes.general_loaded.tube_factory import VesselFactory


class Load(Protocol):
    temperature_delta: float

    def scale(self, factor: float) -> Load:
        ...


class TubeLayer(Protocol):
    def compute_global_strain_vector(
        self, radius: float, solution_vector: np.ndarray, temperature: float
    ) -> np.ndarray:
        ...


class Lamina(Protocol):
    ...


class Vessel(Protocol):
    radii: list[float]
    layers: list[TubeLayer]
    no_of_equations: int
    mid_layer_radii: list[float]

    def create_matrix(self) -> np.ndarray:
        ...

    def create_loading_vector(self, load: Load) -> np.ndarray:
        ...

    def create_force_derivative_matrix(self) -> np.ndarray:
        ...

    def compute_solution_vector(self, load: Load) -> np.ndarray:
        ...


class VesselLayeringStrategy(Protocol):
    ...


class LaminateFactory(Protocol):
    ...


class VesselLayerFactory(Protocol):
    ...


class LayeringStrategy(Protocol):
    ...


class Layer(Protocol):
    thickness: float
    degrees: bool
    rotation: float
    lamina: Lamina


class NoneLinearSolutionMethod(Protocol):
    def compute_solutions(
        self,
        internal_load: callable,
        internal_load_derivative: callable,
        external_load: callable,
        external_load_derivative: callable,
        previous_solution: np.ndarray,
    ) -> np.ndarray:
        ...


class NonLinearThickWalledStrainComputer(ThickWalledStrainComputer):
    def __init__(
        self,
        non_linear_method: NoneLinearSolutionMethod = NewtonRaphsonWithResiduals([ForceResidualConvergence()]),
        steps: int = 4,
        tolerance: float = 1e-3,
        max_iterations: int = int(1e3),
    ) -> None:
        super().__init__(
            VesselFactory(),
        )
        self.non_linear_method = non_linear_method
        self.steps = steps
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def __str__(self) -> str:
        return "Non-Linear Thick-Walled"

    def compute_global_strains(
        self,
        layers: list[Layer],
        load: Load,
        inner_radius: float,
    ) -> np.ndarray:
        try:
            solution = self.compute_solution_vector(layers, load, inner_radius)            
            strains = self.compute_strain_from_solution_vector(
                layers, load, solution, inner_radius
            )
            return strains
        except NegativeDimensionError:
            return None
        except StopIteration:
            return None

    def compute_strain_from_solution_vector(
        self, layers: list[Layer], load: Load, solution: np.ndarray, inner_radius
    ) -> np.ndarray:
        tube = update_tube(
            inner_radius,
            layers,
            solution,
            load.temperature_delta,
            self.vessel_factory,
        )

        return self._compute_global_strains(load, tube, solution)

    def compute_global_step_strains(
        self,
        layers: list[Layer],
        load: Load,
        inner_radius: float,
    ) -> tuple[np.ndarray, list[Layer]]:
        solutions = self.compute_steps_solution_vectors(layers, load, inner_radius)

        loads = scale_loads(load, self.steps)
        strains = np.empty((self.steps + 1, 6, len(layers)))

        for step_index in range(self.steps + 1):
            solution = solutions[:, [step_index]]
            loading_step = loads[step_index]
            tube = update_tube(
                inner_radius,
                layers,
                solution,
                loads[0].temperature_delta,
                self.vessel_factory,
            )
            strains[[step_index], :, :] = self._compute_global_strains(
                loading_step, tube, solution
            )

        return strains

    def compute_steps_solution_vectors(
        self, layers: list[Layer], load: Load, inner_radius: float
    ) -> np.ndarray:
        solutions = compute_solution_vectors(
            inner_radius,
            self.vessel_factory,
            layers,
            load,
            self.non_linear_method,
            self.steps,
            self.tolerance,
            self.max_iterations,
        )
        return solutions

    def compute_solution_vector(
        self, layers: list[Layer], load: Load, inner_radius: float
    ) -> np.ndarray:
        vectors = self.compute_steps_solution_vectors(layers, load, inner_radius)
        return vectors[:, [-1]]


def extract_last_timestep(array: np.ndarray) -> np.ndarray:
    if len(array.shape) == 3:
        return array[-1, :, :]
    return array


def compute_solution_vectors(
    inner_radius: float,
    tube_factory: VesselFactory,
    layers: list[float],
    load: Load,
    non_linear_method: callable,
    steps: int = 1,
    tolerance: float = 1e-3,
    max_iterations: int = int(1e3),
):
    tube: Vessel = tube_factory.create_vessel(inner_radius, layers)
    solutions = np.zeros((tube.no_of_equations, steps + 1))
    
    for i, loading in enumerate(scale_loads(load, steps)[1:]):
        step = NonLinearStep(
            inner_radius,
            layers,
            tube_factory,
            loading,
            non_linear_method,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        solutions[:, [i + 1]] = step.compute_solution_vector(solutions[:, [i]])
    
    return solutions


@dataclass
class NonLinearStep:
    inner_radius: float
    layers: list[Layer]
    vessel_factory: VesselFactory
    loading_step: Load
    non_linear_method: NoneLinearSolutionMethod
    max_iterations: int = int(1e3)
    tolerance: float = 1e-3

    def __post_init__(self):
        self.solutions = list()
        self.tubes = list()

        self.original_tube: Vessel = self.vessel_factory.create_vessel(
            self.inner_radius, self.layers
        )

    def compute_solution_vector(self, previous_solution: np.ndarray):
        return self.non_linear_method.compute_solutions(
            self.internal_load,
            self.internal_load_derivative,
            self.external_load,
            self.external_load_derivative,
            previous_solution,
        )

    def get_tube(self, solution_vector: np.ndarray) -> Vessel:
        return update_tube(
            self.inner_radius,
            self.layers,
            solution_vector,
            self.loading_step.temperature_delta,
            self.vessel_factory,
        )

    def internal_load_derivative(
        self, solution_vector: np.ndarray
    ) -> np.ndarray:
        return self.get_tube(solution_vector).create_matrix()

    def external_load_derivative(
        self, solution_vector: np.ndarray
    ) -> np.ndarray:
        return self.get_tube(solution_vector).create_force_derivative_matrix(
            self.original_tube.radii, self.loading_step, solution_vector
        )

    def internal_load(self, solution_vector: np.ndarray) -> np.ndarray:
        return np.dot(
            self.get_tube(solution_vector).create_matrix(), solution_vector
        )

    def external_load(self, solution_vector: np.ndarray) -> np.ndarray:
        return self.get_tube(solution_vector).create_loading_vector(
            self.loading_step
        )


def compute_timestep(number_of_steps: int) -> float:
    return 1 / number_of_steps


def scale_loads(load: Load, steps: int) -> list[Load]:
    timestep = compute_timestep(steps)
    return [load.scale(i * timestep) for i in range(steps + 1)]


def check_updated_dimension(
    new_dimension: float,
) -> float | NegativeDimensionError:
    if new_dimension >= 0:
        return new_dimension
    raise NegativeDimensionError(new_dimension)


class NegativeDimensionError(ValueError):
    def __init__(self, dimension: float):
        self.dimension = dimension
        self.message = "Negative dimension..."
        super().__init__(self.message)


def update_layer_thickness(
    original_thickness: float, global_strains: np.ndarray
) -> float:
    return check_updated_dimension(
        original_thickness * (1 + global_strains[2])
    )


def update_winding_angle(
    original_winding_angle: float, global_strains: np.ndarray, degrees: bool
) -> float:
    original_winding_angle = (
        np.radians(original_winding_angle)
        if degrees
        else original_winding_angle
    )
    new_angle = np.arctan(
        np.tan(original_winding_angle)
        * (1 + global_strains[1])
        / (1 + global_strains[0])
    )

    return np.degrees(new_angle) if degrees else new_angle


def update_inner_radius(
    original_inner_radius: float, global_strains: np.ndarray
) -> float:
    return check_updated_dimension(
        original_inner_radius * (1 + global_strains[1])
    )


def update_tube(
    inner_radius,
    layers: list[Layer],
    solution_vector: np.ndarray,
    temperature_delta: float,
    vessel_factory: VesselFactory,
) -> Vessel:
    original_vessel: Vessel = vessel_factory.create_vessel(
        inner_radius, layers
    )
    global_strains = [
        tube_layer.compute_global_strain_vector(
            original_vessel.mid_layer_radii[i],
            i,
            solution_vector,
            temperature_delta,
        )
        for i, tube_layer in enumerate(original_vessel.layers)
    ]
    updated_radius = update_inner_radius(inner_radius, global_strains[0])
    updated_thicknesses = [
        update_layer_thickness(layer.thickness, strain)
        for layer, strain in zip(layers, global_strains)
    ]
    
    degrees = [layer.degrees for layer in layers]
    updated_winding_angles = [
        update_winding_angle(layer.rotation, strain, degrees)
        for layer, strain, degrees in zip(layers, global_strains, degrees)
    ]
    materials = [layer.lamina for layer in layers]
    new_layers = create_complete_layers(
        updated_winding_angles, updated_thicknesses, materials, degrees
    )

    return vessel_factory.create_vessel(updated_radius, new_layers)


def main():
    pass


if __name__ == "__main__":
    main()


# End
