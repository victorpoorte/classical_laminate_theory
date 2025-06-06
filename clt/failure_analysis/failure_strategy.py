from __future__ import annotations
 
import copy
from dataclasses import dataclass
from enum import Enum

import numpy as np

from abc import abstractmethod
from typing import Protocol

from pyparsing import Literal

from clt.clt.failure_criteria.puck import FailureIndexResult
from ..layer import Layer


class Load(Protocol):
    magnitude: float
    inner_pressure: float
    def scale(self, factor: float) -> Load:
        ...

class FailureAnalyser(Protocol):
    def compute_critical_factor(self, layers: list[Layer], load: Load) -> float:
        ...

    def compute_failure_indices(self, layers: list[Layer], load: Load) -> np.ndarray:
        ...

    @classmethod
    def _extract_critical_failure_index(cls, indices: np.ndarray) -> FailureIndexResult:
        ...

    @classmethod
    def _determine_critical_index_location(cls, indices: np.ndarray) -> float:
        ...


class Lamina(Protocol):
    ...

class MaterialDegrader(Protocol):
    def degrade_material(self, material: Lamina, failure_indices: np.ndarray) -> Lamina:
        ...


class FailureMode:
    ...

class FailureTypes(Enum):
    FPF = "first_ply_failure"
    LPF = "last_ply_failure"

@dataclass
class FailureResult:
    loads: Load | list[Load]
    modes: FailureMode | list[FailureMode]
    type: Literal[FailureTypes.FPF, FailureTypes.LPF]

    def __post_init__(self):
        self._ensure_loads_list()

    def _ensure_loads_list(self):
        if not isinstance(self.loads, list):
            self.loads: list[Load] = [self.loads]
    
        if not isinstance(self.modes, list):
            self.modes: list[Load] = [self.modes]

    @property
    def is_first_ply(self) -> bool:
        return self.type == FailureTypes.FPF.value

    @property
    def is_last_ply(self) -> bool:
        return self.type == FailureTypes.LPF.value

    @property
    def first_load(self) -> Load:
        return self.loads[0]
    
    @property
    def first_mode(self) -> FailureMode:
        return self.modes[0]
    
    @property
    def last_load(self) -> Load:
        return max(self.loads, key=lambda load: load.magnitude)



class FailureStrategy:
    
    @abstractmethod
    def run(self, layers: list[Layer], load: Load, analyser: FailureAnalyser, max_iterations: int = 40, convergence_margin: float = 0.01) -> FailureResult:
        ...


class FirstPlyFailureStrategy(FailureStrategy):

    def __str__(self):
        return "First Ply Failure"
    
    def run(self, layers: list[Layer], load: Load, analyser: FailureAnalyser, max_iterations: int = 40, convergence_margin: float = 0.01) -> FailureResult:
        for _ in range(max_iterations):
            critical_factor, mode = analyser.compute_critical_factor(layers, load)
            if abs(critical_factor - 1) <= convergence_margin:
                return FailureResult(load, mode, FailureTypes.FPF.value)
            load = load.scale(critical_factor)

        raise ValueError("Exceeded maximum iterations")
    

class LastPlyFailureStrategy(FailureStrategy):

    def __str__(self):
        return "Last Ply Failure"

    def __init__(self, material_degrader: MaterialDegrader) -> None:
        self.material_degrader = material_degrader
    
    def run(self, layers: list[Layer], load: Load, analyser: FailureAnalyser, max_iterations: int = 40, convergence_margin: float = 0.01) -> FailureResult:
        loads: list[Load] = list()
        modes: list[FailureMode] = list()

        # Make a copy of layers, to ensure that these can be degraded
        layers = copy.copy(layers)

        # Compute failure indices
        failure_indices = analyser.compute_failure_indices(layers, load)

        # The initial load should not lead to failure
        if not self._is_valid_initial_load(analyser, failure_indices):
            raise ValueError("Invalid initial load...")

        max_iterations = len(layers) * 4
        for _ in range(max_iterations):

            # Determine if the laminate has failed
            if self._laminate_has_failed(layers, failure_indices):
                return FailureResult(loads, modes, FailureTypes.LPF.value)

            # Scale new load based on factor of safety
            min_factor, mode = analyser.compute_critical_factor(layers, load)
            scaled_load = load.scale(min_factor)

            # Degrade layers till load is redistributed over the layers
            layers = self._degrade_layers_and_redistribute_load(layers, scaled_load, analyser)

            # Store failure progression
            loads.append(scaled_load)
            modes.append(mode)

            # Update load for new operation
            load = scaled_load

            # Compute failure indices and locations
            failure_indices = analyser.compute_failure_indices(layers, load)

    def _degrade_layers_and_redistribute_load(self, layers: list[Layer], load: Load, analyser: FailureAnalyser) -> list[Layer]:
        max_iterations = len(layers) * 2
        for _ in range(max_iterations):

            # Compute new failure indices
            failure_indices = analyser.compute_failure_indices(layers, load)
            index_location = analyser._determine_critical_index_location(failure_indices)
            max_failure_index = analyser._extract_critical_failure_index(failure_indices)

            # Degrade the material of the critical index
            # idx = index_location[0]
            degraded_layer = self._determine_layer_to_degrade_and_degrade(layers, failure_indices, self.material_degrader, analyser)

            # Update layers and laminate with degraded lamina
            layers[index_location] = degraded_layer

            # Means that the current laminate can sustain the load
            if max_failure_index < 1:
                return layers
            
            # Determine if the laminate has failed
            if self._laminate_has_failed(layers, failure_indices):
                return layers

    @staticmethod
    def _laminate_has_failed(layers: list[Layer], indices: list[FailureIndexResult]) -> bool:
        matrix_failures = np.array([layer.lamina.degraded_matrix for layer in layers])
        fibre_failures = np.array([layer.lamina.degraded_fibre for layer in layers])
        failures = np.column_stack((fibre_failures, matrix_failures))

        # Straight forward that all the plies have failed long and transverse
        if np.all(failures):
            return True
        
        # When the critical index has already failed in that direction
        # the laminate is considered as failed
        indices = np.array([
            [index.longitudinal, index.transverse]
            for index in indices
        ])
        max_index = np.unravel_index(np.argmax(indices), indices.shape)

        if failures[max_index]:
            return True
        
        return False
    
    @staticmethod
    def _is_valid_initial_load(analyser: FailureAnalyser, failure_indices: list[FailureIndexResult]) -> bool:
    
        # Compute failure indices and locations
        max_failure_index = analyser._extract_critical_failure_index(failure_indices)



        # The max failure index should be smaller than 1
        return max_failure_index <= 1
    
    @staticmethod
    def _determine_layer_to_degrade_and_degrade(layers: list[Layer], indices: np.ndarray, material_degrader: MaterialDegrader, analyser: FailureAnalyser) -> Layer:
            
        index_location = analyser._determine_critical_index_location(indices)

        # Find index of the layer to be degraded
        # idx = index_location[0]

        # Extract layer material
        layer_to_degrade: Layer = layers[index_location]
        degraded_material = material_degrader.degrade_material(
            layer_to_degrade.lamina, indices[index_location]
        )
        
        return Layer(
            degraded_material,
            layer_to_degrade.thickness,
            layer_to_degrade.rotation,
            layer_to_degrade.degrees
        )


class FailureStrategyInitialiserFactory:

    strategies = {
        FailureTypes.FPF.value: FirstPlyFailureStrategy,
        FailureTypes.LPF.value: LastPlyFailureStrategy,
    }

    @property
    def available(self):
        return ", ".join(self.strategies.keys())

    def get_strategy(self, strategy: str) -> FailureStrategy:
        strat = self.strategies.get(strategy)

        if strat is not None:
            return strat
        
        raise ValueError(
            f"{strategy} not available as analyser"
            + f"\nAvailable analyser are: {self.available}"
        )
    


# End
    