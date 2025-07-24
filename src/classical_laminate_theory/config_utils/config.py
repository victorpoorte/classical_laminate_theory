
from typing import Any, Union
import numpy as np
from dataclasses import dataclass, field

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
        self.layers_builder = LayersBuilder(self.laminate, self.material, self.settings.layering_strategy_factory)

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

    def get_material(self, name: str) -> Lamina:
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
class LaminateStack:
    material_names: list[str]
    angles: list[float]
    layer_thickness: list[float]
    degrees: bool
    layering_strategy: str


@dataclass
class LaminateConfig(Sweepable):
    _config: Union[dict, list[dict]]

    ANGLES: str = "angles"
    SYMMETRIC: str = "symmetric"
    DEGREES: str = "degrees"
    LAYER_THICKNESS: str = "layer_thickness"
    MATERIAL_NAME: str = "material_name"
    LAYERING_STRATEGY: str = "layering_strategy"

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

        # Make make symmetries if needed
        angles = self._handle_angles(cfg)

        # Extract materials and thicknesses, ensure convert to list if needed
        no_layers = len(angles)       
        names = self._to_list(cfg.get(self.MATERIAL_NAME), no_layers)
        layer_thickness = self._to_list(cfg.get(self.LAYER_THICKNESS), no_layers)
        
        # Extract single values for laminate stack
        degrees = cfg.get(self.DEGREES, True)
        layering_strategy = cfg.get(self.LAYERING_STRATEGY)

        return LaminateStack(names, angles, layer_thickness, degrees, layering_strategy)

    def _handle_angles(self, cfg: dict) -> list[float]:
        angles = cfg.get(self.ANGLES)

        if angles is None:
            raise ValueError("Missing 'angles' in laminate config.")

        if isinstance(angles, dict):
            angles = _min_max_step_dict_to_array(angles)

        symmetric = cfg.get(self.SYMMETRIC, 0)

        if symmetric:
            for _ in range(symmetric):
                angles = angles + angles[::-1]
        return angles
    
    @staticmethod
    def _to_list(value, length):
        return value if isinstance(value, list) else [value] * length
    

@dataclass
class LayersBuilder:
    laminate: LaminateConfig
    material: MaterialConfig
    strategy_factory: LayeringStrategyFactory

    @property
    def _strategies(self):
        return [
            self.strategy_factory.create(lam.layering_strategy)
            for lam in self.laminate.values
        ]

    def build(self):
        return [
            strategy.create_complete_layers(
                laminate.angles,
                laminate.layer_thickness,
                [self.material.get_material(mat) for mat in laminate.material_names],
                laminate.degrees 
            )
            for strategy, laminate in zip(self._strategies, self.laminate.values)
        ]


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


class Factory:
    def create(self, item: str) -> Any: ...


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