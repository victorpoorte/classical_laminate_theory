import numpy as np

from .failure_criterion_protocol import (
    FailureCriterion,
    FailureMode,
    StressState,
    Layer,
    FailureStresses,
    classify_failure_mode
)

class FailureCriterion(FailureCriterion):
    name = "Maximum Stress"

    def long_and_trav_index(self, global_strain: np.ndarray, layer: Layer, temperature_delta: float, operating_temperature: float) -> tuple[float, float]:
        
        stress_state = self._compute_layer_stress_state(layer, global_strain, temperature_delta)
        failure_stresses = layer.lamina.get_failure_stress(operating_temperature)
        long = self._longitudinal_index(stress_state, failure_stresses)
        trav = self._traverse_index(stress_state, failure_stresses)

        return long, trav

    def factor_of_safety(self, global_strain: np.ndarray, layer: Layer, temperature_delta: float, operating_temperature: float) -> tuple[float, FailureMode]:

        stress_state = self._compute_layer_stress_state(layer, global_strain, temperature_delta)
        failure_stresses = layer.lamina.get_failure_stress(operating_temperature)
        
        mode = self._define_failure_mode(stress_state, failure_stresses)
        
        fibre_index = self._longitudinal_index(stress_state, failure_stresses)
        matrix_index = self._traverse_index(stress_state, failure_stresses)

        return 1 / max([fibre_index, matrix_index]), mode
    
    def _define_failure_mode(self, stress_state: StressState, failure_stresses: FailureStresses) -> FailureMode:
        long = self._longitudinal_index(stress_state, failure_stresses)
        trav = self._traverse_index(stress_state, failure_stresses)
        failure_type = classify_failure_mode(1 / long, stress_state.longitudinal >= 0, 1 / trav, stress_state.traverse >= 0)
        mode = FailureMode(self.name, failure_type)
        return mode

    @staticmethod
    def _longitudinal_index(
        stress_state: StressState, failure_stresses: FailureStresses
    ) -> float:
        if stress_state.longitudinal > 0:
            failure_stress = failure_stresses.X_t
        else:
            failure_stress = failure_stresses.X_c
        return stress_state.longitudinal / failure_stress

    @staticmethod
    def _traverse_index(
        stress_state: StressState, failure_stresses: FailureStresses
    ) -> float:
        if stress_state.traverse > 0:
            failure_stress = failure_stresses.Y_t
        else:
            failure_stress = failure_stresses.Y_c
        matrix_index = stress_state.traverse / failure_stress
        shear_index = (
            abs(stress_state.shear) / failure_stresses.S
        )

        return max([matrix_index, shear_index])



def main():
    pass


if __name__ == "__main__":
    main()

# End
