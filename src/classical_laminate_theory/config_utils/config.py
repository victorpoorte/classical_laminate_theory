from abc import abstractmethod
from typing import Union
import numpy as np
from dataclasses import dataclass

from ..material import Lamina, LaminaFailureStresses
from ..failure_analysis.failure_strategy import FailureStrategyInitialiserFactory, FailureTypes
from ..failure_criteria.factory import FailureCriteriaFactory
from ..material import MaterialFactory
from ..material_degraders.factory import MaterialDegraderFactory
from ..failure_analysis.failure_analyser import FailureAnalyser
from ..strain_computers import StrainComputerFactory
from ..layering_strategies import LayeringStrategyFactory



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

    def _unpack_settings(self):
        self.settings = SettingsConfig(self._config.get(self.SETTINGS))

    def _unpack_laminate(self):
        self.laminate = LaminateConfig(self._config.get(self.LAMINATE))

    def _unpack_material(self):
        self.material = MaterialConfig(self._config.get(self.MATERIAL))

    def _unpack_loading(self):
        self.loading = LoadingConfig(self._config.get(self.LOADING))

    def create_layers(self):
        return [
            self.settings.layering_strategy.create_complete_layers(
                laminate_stack.angles,
                laminate_stack.thickness,
                self.material.default,
                laminate_stack.degrees
            )
            for laminate_stack in self.laminate.all
        ]


@dataclass
class MaterialConfig(Sweepable):
    _config: dict

    FAILURE = "failure"
    FAILURE_STRESSES = "failure_stresses"

    def __post_init__(self):
        super().__post_init__()
        
        self._materials = [
            self._extract_material(material) for material in self._config
        ]

    @property
    def values(self):
        return self._materials

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
class LaminateStack:
    angles: list[float]
    degrees: bool
    thickness: float

@dataclass
class LaminateConfig(Sweepable):
    _config: Union[dict, list]

    ANGLES: str = "angles"
    SYMMETRIC: str = "symmetric"
    DEGREES: str = "degrees"
    LAYER_THICKNESS: str = "layer_thickness"

    def __post_init__(self):
        super().__post_init__()
        self._laminates = self._build_laminates()

    @property
    def values(self) -> list[LaminateStack]:
        return self._laminates

    def _build_laminates(self) -> list[LaminateStack]:
        # If _config is a list of full laminate dicts
        if isinstance(self._config, list) and all(isinstance(c, dict) for c in self._config):
            return [self._build_single_laminate(cfg) for cfg in self._config]

        # Else assume it's a single laminate definition dict
        return [self._build_single_laminate(self._config)]

    def _build_single_laminate(self, cfg: dict) -> LaminateStack:
        angles = cfg.get(self.ANGLES)

        if angles is None:
            raise ValueError("Missing 'angles' in laminate config.")

        if isinstance(angles, dict):
            angles = _min_max_step_dict_to_array(angles)

        symmetric = cfg.get(self.SYMMETRIC, 0)
        degrees = cfg.get(self.DEGREES, True)
        thickness = cfg.get(self.LAYER_THICKNESS, None)

        if symmetric:
            angles = angles + angles[::-1] * symmetric

        return LaminateStack(angles, degrees, thickness)

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

    failure_criterion_strategy: FailureCriteriaFactory = FailureCriteriaFactory()
    material_factory: MaterialFactory = MaterialFactory()
    layering_factory: LayeringStrategyFactory = LayeringStrategyFactory()
    strain_computer_factory: StrainComputerFactory = StrainComputerFactory()
    material_degrader_factory: MaterialDegraderFactory = MaterialDegraderFactory()
    failure_strategy_initialiser_factory: FailureStrategyInitialiserFactory = FailureStrategyInitialiserFactory()

    FAILURE_CRITERION = "failure_criteria"
    LAYERING_STRATEGY = "layering_strategy"
    STRAIN_COMPUTER = "strain_computers"
    FAILURE_STRATEGIES = "failure_strategies"
    MATERIAL_DEGRADER = "material_degrader"
    RETURN_FAILURE_MODES = "return_failure_modes"
    CASES = "cases"

    def __post_init__(self):
        if self._config is None:
            return None
        
        if self.CASES in self._config:
            self._create_explicit_analysers(self._config[self.CASES])

            return
        
        self._unpack_failure_criteria()
        self._unpack_layering_strategy()
        self._unpack_strain_computers()
        self._unpack_material_degrader()
        self._unpack_failure_strategies()
        self._create_cartesian_analysers()

    def _unpack_failure_criteria(self):
        criteria = self._config.get(self.FAILURE_CRITERION)

        if criteria is None:
            self.failure_criteria = None
            return self.failure_criteria
        
        if isinstance(criteria, str):
            criteria = [criteria]

        self.failure_criteria = [
            self.failure_criterion_strategy.get_failure_criterion(criterion)
            for criterion in criteria
        ]
        
        return self.failure_criteria
    
    def _unpack_layering_strategy(self):
        strategy = self._config.get(self.LAYERING_STRATEGY)

        if strategy is None:
            self.layering_strategy = None
            return self.layering_strategy
        
        self.layering_strategy = self.layering_factory.get_layering_strategy(strategy)

        return self.layering_strategy
    
    def _unpack_strain_computers(self):
        computers = self._config.get(self.STRAIN_COMPUTER)

        if computers is None:
            self.strain_computers = None
            return self.strain_computers

        if isinstance(computers, str):
            computers = [computers]

        self.strain_computers = [
            self.strain_computer_factory.create_strain_computer(computer)
            for computer in computers
        ]

        return self.strain_computers

    def _unpack_failure_strategies(self):
        strategies = self._config.get(self.FAILURE_STRATEGIES)

        if isinstance(strategies, str):
            strategies = [strategies]

        args = [
            [] if strat == FailureTypes.FPF.value
            else [self.material_degrader]
            for strat in strategies
        ]

        self.failure_strategies = [
            self.failure_strategy_initialiser_factory.get_strategy(strat)(*arg)
            for strat, arg in zip(strategies, args)
        ]

        return self.failure_strategies

    def _unpack_material_degrader(self):
        degrader = self._config.get(self.MATERIAL_DEGRADER)

        if degrader is None:
            self.material_degrader = None
            return self.material_degrader
        
        self.material_degrader = self.material_degrader_factory.get_degrader(degrader)

        return self.material_degrader

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
        self.analysers = []

        for case in case_dicts:
            criterion = self.failure_criterion_strategy.get_failure_criterion(case[self.FAILURE_CRITERION])
            computer = self.strain_computer_factory.create_strain_computer(case[self.STRAIN_COMPUTER])
            strategy_name = case[self.FAILURE_STRATEGIES]
            self.layering_strategy = self.layering_factory.get_layering_strategy(case[self.LAYERING_STRATEGY])

            degrader = None
            if self.MATERIAL_DEGRADER in case:
                degrader = self.material_degrader_factory.get_degrader(case[self.MATERIAL_DEGRADER])

            if strategy_name == FailureTypes.FPF.value:
                strategy = self.failure_strategy_initialiser_factory.get_strategy(strategy_name)()
            else:
                strategy = self.failure_strategy_initialiser_factory.get_strategy(strategy_name)(degrader)

            self.analysers.append(
                FailureAnalyser(computer, criterion, strategy)
            )

# End