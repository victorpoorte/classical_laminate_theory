import numpy as np


from dataclasses import dataclass
from classical_laminate_theory import Layer, Lamina, LaminaFailureStresses
from classical_laminate_theory.clt.failure_analysis.failure_strategy import FailureStrategyInitialiserFactory, FailureTypes
from classical_laminate_theory.clt.failure_criteria.factory import FailureCriteriaFactory
from classical_laminate_theory.clt.material import MaterialFactory
from classical_laminate_theory.clt.material_degraders.factory import MaterialDegraderFactory
from classical_laminate_theory.clt.failure_analysis.failure_analyser import FailureAnalyser
from common.layering_strategies import LayeringStrategyFactory
from strain_computers.computer_factory import StrainComputerFactory

FAILURE_CRITERION_FACTORY = FailureCriteriaFactory()
MATERIAL_FACTORY = MaterialFactory()
LAYERING_FACTORY = LayeringStrategyFactory()
COMPUTER_FACTORY = StrainComputerFactory()
MATERIAL_DEGRADER_FACTORY = MaterialDegraderFactory()
FAILURE_STRATEGY_FACTORY = FailureStrategyInitialiserFactory()


def _min_max_step_dict_to_array(values: dict) -> np.ndarray:
    return np.arange(
        values["min"], values["max"] + values["step"], values["step"]
    )

@dataclass
class CLTConfig:
    _config: dict

    LOADING = "loading"
    LAMINATE = "laminate"
    MATERIAL = "material"
    SETTINGS = "settings"

    def __post_init__(self):
        self.material = MaterialConfig(self._config.get(self.MATERIAL))
        self.laminate = LaminateConfig(self._config.get(self.LAMINATE))
        self._unpack_loading()
        self.settings = SettingsConfig(self._config.get(self.SETTINGS))

    def _unpack_loading(self):
        self.loading = LoadingConfig(self._config.get(self.LOADING))

    def create_layers(self):
        return self.settings.layering_strategy.create_complete_layers(
            self.laminate.angles,
            self.laminate.layer_thickness,
            self.material.material,
            self.laminate.degrees
        )


@dataclass
class MaterialConfig:
    _config: dict

    FAILURE = "failure"
    FAILURE_STRESSES = "failure_stresses"

    def __post_init__(self):
        self._extract_material()

    def _extract_material(self):
        if self._config is None:
            self.material = None
            return self.material

        if isinstance(self._config, str):
            self.material = MATERIAL_FACTORY.create_material(self._config)
            return self.material

        self._config: dict
        failure_stresses = None
        if self.FAILURE in self._config:
            failure_loads = self._config.pop(self.FAILURE)
            if isinstance(failure_loads, dict):
                failure_stresses = LaminaFailureStresses(**failure_loads)
            elif isinstance(failure_loads, list):
                failure_stresses  = {
                    loads["temperature"]: LaminaFailureStresses(**loads[self.FAILURE])
                    for loads in failure_loads
                }
        self.material = Lamina(**(self._config | {self.FAILURE_STRESSES: failure_stresses}))

        return self.material


@dataclass
class LaminateConfig:
    _config: dict

    ANGLES: str = "angles"
    SYMMETRIC: str = "symmetric"
    DEGREES: str = "degrees"
    LAYER_THICKNESS: str = "layer_thickness"

    def __post_init__(self):
        self._extract_angles()
        self.symmetric = self._config.get(self.SYMMETRIC)
        self.degrees = self._config.get(self.DEGREES)
        self.layer_thickness = self._config.get(self.LAYER_THICKNESS)

        self._create_laminate_symmetries()

    def _extract_angles(self):
        self.angles = self._config.get(self.ANGLES)

        if self.angles is None:
            return self.angles

        if isinstance(self.angles, list):
            return self.angles
        
        if isinstance(self.angles, dict):
            self.angles = _min_max_step_dict_to_array(self.angles)

            return self.angles
        
        raise ValueError(f"Unsupported laminate angles: {self.angles}")

    def _create_laminate_symmetries(self):
        if self.symmetric is not None:
            for _ in range(self.symmetric):
                self.angles += self.angles[::-1]


@dataclass
class FailureEnvelopeConfig:
    name: str
    x_axis: str
    y_axis: str
    angle_resolution: float


@dataclass
class LoadingConfig:
    _config: list[dict]

    def __post_init__(self):
        if self._config is None:
            return None
        
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

    FAILURE_CRITERION = "failure_criteria"
    LAYERING_STRATEGY = "layering_strategy"
    STRAIN_COMPUTER = "strain_computers"
    FAILURE_STRATEGIES = "failure_strategies"
    MATERIAL_DEGRADER = "material_degrader"
    RETURN_FAILURE_MODES = "return_failure_modes"

    def __post_init__(self):
        if self._config is None:
            return None
        
        self._unpack_failure_criteria()
        self._unpack_layering_strategy()
        self._unpack_strain_computers()
        self._unpack_material_degrader()
        self._unpack_failure_strategies()

        self.return_failure_modes = self._config.get(self.RETURN_FAILURE_MODES)

        self._create_analysers()

    def _unpack_failure_criteria(self):
        criteria = self._config.get(self.FAILURE_CRITERION)

        if criteria is None:
            self.failure_criteria = None
            return self.failure_criteria
        
        if isinstance(criteria, str):
            criteria = [criteria]

        self.failure_criteria = [
            FAILURE_CRITERION_FACTORY.get_failure_criterion(criterion)
            for criterion in criteria
        ]
        
        return self.failure_criteria
    
    def _unpack_layering_strategy(self):
        strategy = self._config.get(self.LAYERING_STRATEGY)

        if strategy is None:
            self.layering_strategy = None
            return self.layering_strategy
        
        self.layering_strategy = LAYERING_FACTORY.get_layering_strategy(strategy)

        return self.layering_strategy
    
    def _unpack_strain_computers(self):
        computers = self._config.get(self.STRAIN_COMPUTER)

        if computers is None:
            self.strain_computers = None
            return self.strain_computers

        if isinstance(computers, str):
            computers = [computers]

        self.strain_computers = [
            COMPUTER_FACTORY.create_strain_computer(computer)
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
            FAILURE_STRATEGY_FACTORY.get_strategy(strat)(*arg)
            for strat, arg in zip(strategies, args)
        ]

        return self.failure_strategies

    def _unpack_material_degrader(self):
        degrader = self._config.get(self.MATERIAL_DEGRADER)

        if degrader is None:
            self.material_degrader = None
            return self.material_degrader
        
        self.material_degrader = MATERIAL_DEGRADER_FACTORY.get_degrader(degrader)

        return self.material_degrader

    def _create_analysers(self):
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


# End