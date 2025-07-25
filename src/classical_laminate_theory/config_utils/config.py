
from typing import Any, Union
import numpy as np
from dataclasses import dataclass, field

from ..layer import LayersBuilder
from ..laminate import LaminateStack
from ..material import Lamina, LaminaFailureStresses
from ..failure_analysis.failure_strategy import FailureStrategyInitialiserFactory
from ..failure_criteria.factory import FailureCriteriaFactory
from ..material import MaterialFactory
from ..material_degraders.factory import MaterialDegraderFactory
from ..failure_analysis.failure_analyser import FailureAnalyser
from ..strain_computers import StrainComputerFactory
from ..layering_strategies import LayeringStrategyFactory


class Factory:
    def create(self, item: str) -> Any: ...


def _min_max_step_dict_to_array(values: dict) -> np.ndarray:
    return np.arange(
        values["min"], values["max"] + values["step"], values["step"]
    )



@dataclass
class Sweepable:
    _config: Union[dict, list, float]

    def __post_init__(self):
        self._config = self._config if isinstance(self._config, list) else [self._config]

    def is_sweep(self):
        return len(self._config) > 1
    
    @property
    def values(self):
        return self._config

    @property
    def default(self):
        if self.is_sweep():
            raise ValueError("Sweep defined, single value not accessible")
        return self.values[0]

    @property
    def all(self):
        return self.values


@dataclass
class CLTConfig:
    _config: dict

    LOADING = "loading"
    LAMINATE = "laminate"
    MATERIAL = "material"
    SETTINGS = "settings"

    def __post_init__(self):
        self._unpack_material()
        self._unpack_laminate()
        self._unpack_loading()
        self._unpack_settings()
        self._create_builder()

    def _unpack_settings(self):
        self.settings = SettingsConfig(self._config.get(self.SETTINGS))

        return self.settings

    def _unpack_laminate(self):
        self.laminate = LaminateConfig(self._config.get(self.LAMINATE))

        return self.laminate

    def _unpack_material(self):
        self.material = MaterialConfig(self._config.get(self.MATERIAL))

        return self.material

    def _unpack_loading(self):
        self.loading = LoadingConfig(self._config.get(self.LOADING))

        return self.loading
    
    def _create_builder(self):
        self.layers_builder = LayersBuilder(
            self.laminate.values, 
            self.material,
            self.settings.layering_strategy_factory
        )

        return self.layers_builder


@dataclass
class MaterialConfig(Sweepable):
    _config: dict

    FAILURE = "failure"
    FAILURE_STRESSES = "failure_stresses"
    NAME = "name"

    def __post_init__(self):
        super().__post_init__()
        
        self._materials = {
            material if isinstance(material, str) else material[self.NAME]:
            self._extract_material(material)
            for material in self._config
        }

    @property
    def values(self):
        return self._materials
    
    @property
    def default(self) -> Lamina:
        if len(self._materials) != 1:
            raise ValueError("MaterialConfig has multiple materials, cannot pick default.")
        return list(self._materials.values())[0]

    def get(self, name: str) -> Lamina:
        material = self._materials.get(name)
        if material is None:
            raise ValueError(
                f"Material '{name}' not available in MaterialConfig\n"
                + f"Available materials are: {", ".join(self._materials.keys())}"
            )
        return material

    def _extract_material(self, material: Union[None, str, dict]):
        if material is None:
            return None

        if isinstance(material, str):
            return MaterialFactory().create_material(material)

        material: dict
        failure_stresses = None
        if self.FAILURE in material:
            failure_loads = material.pop(self.FAILURE)
            if isinstance(failure_loads, dict):
                failure_stresses = LaminaFailureStresses(**failure_loads)
            elif isinstance(failure_loads, list):
                failure_stresses  = {
                    loads["temperature"]: LaminaFailureStresses(**loads[self.FAILURE])
                    for loads in failure_loads
                }
        
        return Lamina(**(material | {self.FAILURE_STRESSES: failure_stresses}))


@dataclass
class LaminateConfig(Sweepable):
    _config: Union[dict, list[dict]]

    ANGLES: str = "angles"
    SYMMETRIC: str = "symmetric"
    DEGREES: str = "degrees"
    LAYER_THICKNESS: str = "layer_thicknesses"
    MATERIAL_NAME: str = "material_names"
    LAYERING_STRATEGY: str = "layering_strategy"

    def __post_init__(self):
        super().__post_init__()
        self._laminates = self._build_laminates()

    @property
    def values(self) -> list[LaminateStack]:
        return self._laminates

    @property
    def stack_builders(self) -> dict[str, callable]:
        return {
            "sweep": self._build_sweep_stacks,
            "list": self._build_list_of_orientations_stacks,
            "single": self._build_simple_orientations_stack,
        }

    def _build_laminates(self) -> list[LaminateStack]:

        all_stacks = list()
        for laminate in self._config:

            laminate = laminate.copy()
            angles = laminate.pop(self.ANGLES)
            symmetric = laminate.pop(self.SYMMETRIC, 0)
            mode = self._determine_lamination_mode(angles)
            stacks = self.stack_builders[mode](laminate, angles, symmetric)
            all_stacks.extend(LaminateStack(**stack) for stack in stacks)
        
        return all_stacks

    def _build_simple_orientations_stack(self, laminate, angles, symmetric):
        angles = self._handle_symmetries(angles, symmetric)
        laminate = self._normalise_laminate_metadata(laminate, angles)
        stacks = [{**laminate, self.ANGLES: angles}]
        return stacks

    def _build_list_of_orientations_stacks(self, laminate, angles, symmetric):
        stacks = list()
        for angle in angles:
            angle = self._handle_symmetries(angle, symmetric)
            laminate = self._normalise_laminate_metadata(laminate.copy(), angle)
            stacks.append({**laminate, self.ANGLES: angle})
        return stacks

    def _build_sweep_stacks(self, laminate, angles, _):
        return [
            {**self._normalise_laminate_metadata(laminate, [angle]), self.ANGLES: [angle]}
            for angle in _min_max_step_dict_to_array(angles)
        ]
    
    def _normalise_laminate_metadata(self, laminate: dict, angles: list[float]) -> dict:
        number_of_layers = len(angles)

        for value in [self.MATERIAL_NAME, self.LAYER_THICKNESS]:
            laminate[value] = self._to_list(laminate[value], number_of_layers)


        return laminate
    
    def _handle_symmetries(self, angles, symmetric):
        for _ in range(symmetric):
            angles += angles[::-1]
        
        return angles

    def _are_simple_orientations(self, angles):
        return isinstance(angles, list) and (isinstance(angles[0], int) or isinstance(angles[0], float))

    def _is_list_of_orientations(self, angles):
        return isinstance(angles[0], list) and (isinstance(angles[0][0], float) or isinstance(angles[0][0], int))

    def _is_sweep_laminate(self, angles):
        return isinstance(angles, dict)
    
    @staticmethod
    def _to_list(value, length):
        if isinstance(value, list):
            if len(value) != length:
                raise ValueError(f"Expected list of length {length}, got {len(value)}")
            return value
        return [value] * length    

    def _determine_lamination_mode(self, angles: list) -> str:
        if self._is_sweep_laminate(angles):
            return "sweep"
        if self._is_list_of_orientations(angles):
            return "list"
        if self._are_simple_orientations(angles):
            return "single"
        
        raise ValueError("Unsupported laminate definition...")

    def get_sweep_angles(self):
        return [lam.angles for lam in self.all]
    
    def get_sweep_thicknesses(self):
        thicknesses = self.default.layer_thicknesses
        if len(thicknesses) != 1:
            raise ValueError("Not suported thickness")
        thickness = thicknesses[0]
        if isinstance(thickness, float) or isinstance(thickness, int):
            return thickness
        return _min_max_step_dict_to_array(thickness)
    
    def get_sweep_material_name(self):
        names = self.default.material_names
        if len(names) != 1:
            raise ValueError("get_material_name invalid method for defined materials names")
        return names[0]
    
    def get_sweep_layering_strategy(self):
        return self.default.layering_strategy
    
    def get_sweep_degrees(self):
        return self.default.degrees

@dataclass
class FailureEnvelopeConfig:
    name: str
    x_axis: str
    y_axis: str
    angle_resolution: float


class LoadingConfig(Sweepable):

    def __post_init__(self):
        super().__post_init__()
        
        self._unpack_loads()

    def _unpack_loads(self):
        self.loads = [
            FailureEnvelopeConfig(**load)
            for load in self._config
        ]

        return self.loads



@dataclass
class SettingsConfig:
    _config: dict

    failure_criterion_factory: Factory = field(default_factory=FailureCriteriaFactory)
    layering_strategy_factory: Factory = field(default_factory=LayeringStrategyFactory)
    strain_computer_factory: Factory = field(default_factory=StrainComputerFactory)
    failure_strategy_factory: Factory = field(default_factory=FailureStrategyInitialiserFactory)
    material_degrader_factory: Factory = field(default_factory=MaterialDegraderFactory)

    FAILURE_CRITERION = "failure_criteria"
    LAYERING_STRATEGY = "layering_strategy"
    STRAIN_COMPUTER = "strain_computers"
    FAILURE_STRATEGIES = "failure_strategies"
    MATERIAL_DEGRADER = "material_degrader"
    CASES = "cases"

    def __post_init__(self):

        self._create_factory_map()

        if self._config is None:
            return None
        
        if self.CASES in self._config:
            self._create_explicit_analysers(self._config[self.CASES])

            return
        
        self._unpack_failure_criteria()
        self._unpack_strain_computers()
        self._unpack_material_degrader()
        self._unpack_failure_strategies()
        self._create_cartesian_analysers()

    def _create_factory_map(self):
        self.factory_map: dict[str, Factory] = {
            self.FAILURE_CRITERION: self.failure_criterion_factory,
            self.LAYERING_STRATEGY: self.layering_strategy_factory,
            self.STRAIN_COMPUTER: self.strain_computer_factory,
            self.FAILURE_STRATEGIES: self.failure_strategy_factory,
            self.MATERIAL_DEGRADER: self.material_degrader_factory,
        }

        return self.factory_map

    def _unpack_item(self, items: str, factory: Factory):
        if items is None:
            return None
        
        if isinstance(items, str):
            items = [items]

        return [factory.create(item) for item in items]

    def _unpack_failure_criteria(self):
        self.failure_criteria = self._unpack_item(
            self._config.get(self.FAILURE_CRITERION),
            self.factory_map[self.FAILURE_CRITERION]
        )

        return self.failure_criteria
    
    def _unpack_strain_computers(self):
        self.strain_computers = self._unpack_item(
            self._config.get(self.STRAIN_COMPUTER),
            self.factory_map[self.STRAIN_COMPUTER]
        )

        return self.strain_computers

    def _unpack_material_degrader(self):
        self.material_degraders = self._unpack_item(
            self._config.get(self.MATERIAL_DEGRADER),
            self.factory_map[self.MATERIAL_DEGRADER]
        )

        return self.material_degraders

    def _unpack_failure_strategies(self):
        strategies = self._config.get(self.FAILURE_STRATEGIES)

        if isinstance(strategies, str):
            strategies = [strategies]

        degraders = self.material_degraders if self.material_degraders is not None else [None] * len(strategies)
        self.failure_strategies = [
            self.factory_map[self.FAILURE_STRATEGIES].create(strat)(degrader)
            for strat, degrader in zip(strategies, degraders)
        ]

        return self.failure_strategies

    def _create_cartesian_analysers(self):
        self.analysers = [
            FailureAnalyser(computer, criterion, failure_strategy)
            for criterion in self.failure_criteria
            for computer in self.strain_computers
            for failure_strategy in self.failure_strategies
        ]

        return self.analysers
    
    def _create_analyser_labels(self, computer: str, criterion: str, failure_strategy: str) -> str:
        labels = list()
        if len(self.strain_computers) > 1:
            labels.append(computer)
        if len(self.failure_criteria) > 1:
            labels.append(criterion)
        if len(self.failure_strategies) > 1:
            labels.append(failure_strategy)

        return " - ".join(labels)

    def _create_explicit_analysers(self, case_dicts: list[dict]):
        self.analysers = [
            self._case_to_failure_analyser(case) for case in case_dicts
        ]

        return self.analysers
    
    def _case_to_failure_analyser(self, case: dict) -> FailureAnalyser:

        material_degrader = (
            None if self.MATERIAL_DEGRADER not in case
            else self.factory_map[self.MATERIAL_DEGRADER].create(case[self.MATERIAL_DEGRADER])
        )
        strain_computer = self.factory_map[self.STRAIN_COMPUTER].create(case[self.STRAIN_COMPUTER])
        failure_criterion = self.factory_map[self.FAILURE_CRITERION].create(case[self.FAILURE_CRITERION])
        failure_strategy = self.factory_map[self.FAILURE_STRATEGIES].create(case[self.FAILURE_STRATEGIES])(material_degrader)
        
        return FailureAnalyser(strain_computer, failure_criterion, failure_strategy)

# End