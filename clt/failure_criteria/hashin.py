import math

import numpy as np

from .failure_criterion_protocol import (
    FailureCriterion,
    FailureMode,
    FailureStresses,
    StressState,
    Layer,
    classify_failure_mode
)


class FailureCriterion(FailureCriterion):
    alpha = 1.0
    name = "Hashin"

    def long_and_trav_index(self, global_strain: np.ndarray, layer: Layer, temperature_delta: float, operating_temperature: float) -> tuple[float, float]:
        long, trav = self._long_and_trav_factor(global_strain, layer, temperature_delta, operating_temperature)

        return 1 / long, 1 / trav

    def factor_of_safety(self, global_strain: np.ndarray, layer: Layer, temperature_delta: float, operating_temperature: float) -> tuple[float, FailureMode]:

        stress_state = self._compute_layer_stress_state(layer, global_strain, temperature_delta)
        failure_stresses = layer.lamina.get_failure_stress(operating_temperature)
        mode = self._define_failure_mode(stress_state, failure_stresses)

        longitudinal, traverse = self._long_and_trav_factor(
            global_strain, layer, temperature_delta, operating_temperature
        )

        return min([longitudinal, traverse]), mode

    def _define_failure_mode(self, stress_state: StressState, failure_stresses: FailureStresses) -> FailureMode:
        long = self._longitudinal_factor(stress_state, failure_stresses)
        trav = self._traverse_factor(stress_state, failure_stresses)
        failure_type = classify_failure_mode(long, stress_state.longitudinal >= 0, trav, stress_state.traverse >= 0)
        mode = FailureMode(self.name, failure_type)
        return mode

    def _long_and_trav_factor(self, global_strain: np.ndarray, layer: Layer, temperature_delta: float, operating_temperature: float) -> tuple[float, float]:
        stress_state = self._compute_layer_stress_state(
            layer, global_strain, temperature_delta
        )
        failure_stresses = layer.lamina.get_failure_stress(operating_temperature)
        longitudinal = self._longitudinal_factor(stress_state,  failure_stresses)
        traverse = self._traverse_factor(stress_state,  failure_stresses)
        
        return longitudinal, traverse

    def _longitudinal_factor(
        self,
        stress_state: StressState,
        failure_stresses: FailureStresses,
    ) -> float:
        if self._tension(stress_state.longitudinal):
            return self._longitudinal_tension_factor_of_safety(
                stress_state, failure_stresses
            )
        return self._longitudinal_compression_factor_of_safety(
            stress_state, failure_stresses
        )

    def _traverse_factor(
        self,
        stress_state: StressState,
        failure_stresses: FailureStresses,
    ) -> float:
        if self._tension(stress_state.traverse):
            return self._traverse_tension_factor_of_safety(stress_state, failure_stresses)
        return self._traverse_compression_factor_of_safety(stress_state, failure_stresses)

    def _longitudinal_tension_factor(
        self, stress_state: StressState, failure_stresses: FailureStresses
    ) -> float:
        return (
            stress_state.longitudinal / failure_stresses.X_t
        ) ** 2 + self.alpha * (
            stress_state.shear / failure_stresses.S
        ) ** 2

    def _longitudinal_compression_factor(
        self, stress_state: StressState, failure_stresses: FailureStresses
    ) -> float:
        return (stress_state.longitudinal / failure_stresses.X_c) ** 2

    def _traverse_tension_factor(
        self,
        stress_state: StressState,
        failure_stresses: FailureStresses,
    ) -> float:
        return (
            stress_state.traverse / failure_stresses.Y_t
        ) ** 2 + (
            stress_state.shear / failure_stresses.S
        ) ** 2

    def _traverse_compression_factor(
        self,
        stress_state: StressState,
        failure_stresses: FailureStresses,
    ) -> float:
        return (
            (stress_state.traverse / (2 * failure_stresses.S))
            ** 2
            + (
                (
                    (
                        failure_stresses.Y_c
                        / (2 * failure_stresses.S)
                    )
                    ** 2
                    - 1
                )
                * (
                    stress_state.traverse
                    / failure_stresses.Y_c
                )
            )
            + (stress_state.shear / failure_stresses.S) ** 2
        )

    def _longitudinal_tension_factor_of_safety(
        self, stress_state: StressState, failure_stresses: FailureStresses
    ) -> float:
        return math.sqrt(
            1 / self._longitudinal_tension_factor(stress_state, failure_stresses)
        )

    def _longitudinal_compression_factor_of_safety(
        self, stress_state: StressState, failure_stresses: FailureStresses
    ) -> float:
        return math.sqrt(
            1
            / self._longitudinal_compression_factor(
                stress_state, failure_stresses
            )
        )

    def _traverse_tension_factor_of_safety(
        self,
        stress_state: StressState,
        failure_stresses: FailureStresses,
    ) -> float:
        return math.sqrt(
            1 / self._traverse_tension_factor(stress_state, failure_stresses)
        )

    def _determine_a_factor_of_safety(
        self,
        stress_state: StressState,
        failure_stresses: FailureStresses,
    ) -> float:
        return (
            stress_state.traverse / (2 * failure_stresses.S)
        ) ** 2 + (
            stress_state.shear / failure_stresses.S
        ) ** 2

    def _determine_b_factor_of_safety(
        self,
        stress_state: StressState,
        failure_stresses: FailureStresses,
    ) -> float:
        return (
            (
                failure_stresses.Y_c
                / (2 * failure_stresses.S)
            )
            ** 2
            - 1
        ) * (stress_state.traverse / failure_stresses.Y_c)

    def _traverse_compression_factor_of_safety(
        self, stress_state: StressState, failure_stresses: FailureStresses
    ) -> float:
        a = self._determine_a_factor_of_safety(stress_state, failure_stresses)
        b = self._determine_b_factor_of_safety(stress_state, failure_stresses)

        return (math.sqrt(b**2 + 4 * a) - b) / (2 * a)


def main():
    pass


if __name__ == "__main__":
    main()

# End
