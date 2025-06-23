import math

import numpy as np

from .puck import FailureIndexResult
from .failure_criterion_protocol import (
    FailureCriterion,
    FailureMode,
    FailureStresses,
    Layer,
    StressState,
    classify_failure_mode,
)


class FailureCriterion(FailureCriterion):
    name = "Tsai-Wu"


    def factor_of_safety(
        self, global_strain: np.ndarray, layer: Layer, temperature_delta: float, operating_temperature: float
    ) -> float:
        

        stress_state = self._compute_layer_stress_state(
            layer, global_strain, temperature_delta
        )
        failure_stresses = layer.lamina.get_failure_stress(operating_temperature)
        a = self._compute_A(stress_state, failure_stresses)
        b = self._compute_B(stress_state, failure_stresses)
        failure_index = self.long_and_trav_index(global_strain, layer, temperature_delta, operating_temperature)
        return (math.sqrt(b**2 + 4 * a) - b) / (2 * a), failure_index
    
    def _define_failure_mode(self, stress_state: StressState, failure_stresses: FailureStresses) -> FailureMode:
        long = self._longitudinal_index(stress_state, failure_stresses)
        trav = self._traverse_index(stress_state, failure_stresses)
        failure_type = classify_failure_mode(1 / long, stress_state.longitudinal >= 0, 1 / trav, stress_state.traverse >= 0)
        mode = FailureMode(self.name, failure_type)
        return mode

    def long_and_trav_index(self, global_strain: np.ndarray, layer: Layer, temperature_delta: float, operating_temperature: float) -> tuple[float, float]:
        stress_state = self._compute_layer_stress_state(
            layer, global_strain, temperature_delta
        )
        failure_stresses = layer.lamina.get_failure_stress(operating_temperature)
        long = self._longitudinal_index(stress_state, failure_stresses)
        trav = self._traverse_index(stress_state, failure_stresses)

        failure_index = FailureIndexResult(long, trav)
        return failure_index

    def _longitudinal_index(
        self, stress_state: StressState, failure_stresses: FailureStresses
    ) -> float:
        return (
            self._compute_F1(failure_stresses) * stress_state.longitudinal
            + self._compute_F11(failure_stresses)
            * stress_state.longitudinal**2
        )

    def _traverse_index(
        self, stress_state: StressState, failure_stresses: FailureStresses
    ) -> float:
        return (
            self._compute_F2(failure_stresses) * stress_state.traverse
            + self._compute_F22(failure_stresses) * stress_state.traverse**2
            + self._compute_F66(failure_stresses) * stress_state.shear**2
            - self._compute_F11(failure_stresses)
            * stress_state.longitudinal
            * stress_state.traverse
        )

    @staticmethod
    def _compute_F1(failure_stresses: FailureStresses) -> float:
        return (
            1 / failure_stresses.X_t
            + 1 / failure_stresses.X_c
        )

    @staticmethod
    def _compute_F2(failure_stresses: FailureStresses) -> float:
        return (
            1 / failure_stresses.Y_t
            + 1 / failure_stresses.Y_c
        )

    @staticmethod
    def _compute_F11(failure_stresses: FailureStresses) -> float:
        return -1 / (
            failure_stresses.X_t
            * failure_stresses.X_c
        )

    @staticmethod
    def _compute_F22(failure_stresses: FailureStresses) -> float:
        return -1 / (
            failure_stresses.Y_t
            * failure_stresses.Y_c
        )

    @staticmethod
    def _compute_F66(failure_stresses: FailureStresses) -> float:
        return 1 / failure_stresses.S**2

    @staticmethod
    def _compute_Fij(
        Fi: float,
        Fj: float,
        Fii: float,
        Fjj: float,
        failure_stresses: FailureStresses,
    ) -> float:
        # return 0
        return -0.5 * Fii
        return -0.5 * math.sqrt(Fii * Fjj)

    @classmethod
    def _compute_F12(cls, failure_stresses: FailureStresses) -> float:
        return cls._compute_Fij(
            cls._compute_F1(failure_stresses),
            cls._compute_F2(failure_stresses),
            cls._compute_F11(failure_stresses),
            cls._compute_F22(failure_stresses),
            failure_stresses,
        )

    @classmethod
    def _compute_A(
        cls, stress_state: StressState, failure_stresses: FailureStresses
    ) -> float:
        return (
            cls._compute_F11(failure_stresses) * stress_state.longitudinal**2
            + cls._compute_F22(failure_stresses) * stress_state.traverse**2
            + cls._compute_F66(failure_stresses) * stress_state.shear**2
            + 2
            * cls._compute_F12(failure_stresses)
            * stress_state.longitudinal
            * stress_state.traverse
        )

    @classmethod
    def _compute_B(
        cls, stress_state: StressState, failure_stresses: FailureStresses
    ) -> float:
        return (
            cls._compute_F1(failure_stresses) * stress_state.longitudinal
            + cls._compute_F2(failure_stresses) * stress_state.traverse
        )

def main():
    pass


if __name__ == "__main__":
    main()


# End
