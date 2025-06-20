import math
import numpy as np

from .failure_criterion_protocol import (
    FailureCriterion,
    FailureMode,
    FailureStresses,
    FailureType,
    StressState,
    Layer,
    classify_failure_mode,
)


class FailureCriterion(FailureCriterion):
    name = "Tsai-Wu Thick"

    def factor_of_safety(self, global_strain: np.ndarray, layer: Layer, temperature_delta: float, operating_temperature: float) -> float:

        stress_state = self._compute_layer_stress_state(
            layer, global_strain, temperature_delta
        )
        failure_stresses = layer.lamina.get_failure_stress(operating_temperature)

        mode = self._define_failure_mode(stress_state, failure_stresses)

        a = self._compute_A(stress_state, failure_stresses)
        b = self._compute_B(stress_state, failure_stresses)
        try:
            return (math.sqrt(b**2 + 4 * a) - b) / (2 * a), mode
        except ValueError:
            if (b**2 + 4 * a) < 1e-5:
                return -b / (2 * a), mode
        raise ValueError("Negative square root")

    def _define_failure_mode(self, stress_state, failure_stresses):
        long = self._longitudinal_index(stress_state, failure_stresses)
        trav = self._traverse_index(stress_state, failure_stresses)
        failure_type = classify_failure_mode(1 / long, stress_state.longitudinal >= 0, 1 / trav, stress_state.traverse >= 0)
        mode = FailureMode(self.name, failure_type)
        return mode

    def long_and_trav_index(self, global_strain: np.ndarray, layer: Layer, temperature_delta: float, operating_temperature: float) -> tuple[float, float]:
        stress_state = self._compute_layer_stress_state(layer, global_strain, temperature_delta)
        failure_stresses = layer.lamina.get_failure_stress(operating_temperature)
        long = self._longitudinal_index(stress_state, failure_stresses)
        trav = self._traverse_index(stress_state, failure_stresses)

        return long, trav

    def _longitudinal_index(
        self,
        stress_state: StressState,
        failure_stresses: FailureStresses,
    ) -> float:
        return (
            self._compute_F1(failure_stresses) * stress_state.longitudinal
            + self._compute_F11(failure_stresses)
            * stress_state.longitudinal**2
        )

    def _traverse_index(
        self,
        stress_state: StressState,
        failure_stresses: FailureStresses,
    ) -> float:
        if self._thick_stress_state(stress_state):
            self._thick_traverse_index(stress_state, failure_stresses)
        return self._thin_traverse_index(stress_state, failure_stresses)

    def _thick_traverse_index(
        self,
        stress_state: StressState,
        failure_stresses: FailureStresses,
    ) -> float:
        f_vector = self._f_vector(failure_stresses)
        F_matrix = self._F_matrix(failure_stresses)
        stress_vector = stress_state.vector
        matrix_terms = [
            F_matrix[i, j] * stress_vector[j, 0] * stress_vector[i, 0]
            for i in range(6)
            for j in range(6)
            if i != 0 or j != 0
        ]
        vector_terms = f_vector[1:, 0] * stress_vector[1:, 0]

        return sum(matrix_terms) + sum(vector_terms)

    def _thin_traverse_index(
        self,
        stress_state: StressState,
        failure_stresses: FailureStresses,
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
    def _thick_stress_state(stress_state: StressState) -> bool:
        return stress_state.vector[2] != 0

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

    @classmethod
    def _compute_F3(cls, failure_stresses: FailureStresses) -> float:
        return cls._compute_F2(failure_stresses)

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

    @classmethod
    def _compute_F33(cls, failure_stresses: FailureStresses) -> float:
        return cls._compute_F22(failure_stresses)

    @staticmethod
    def _compute_F44(failure_stresses: FailureStresses) -> float:
        return 1 / failure_stresses.S**2

    @staticmethod
    def _compute_F55(failure_stresses: FailureStresses) -> float:
        return 1 / failure_stresses.S**2

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
        # No interaction
        # return 0

        # Quadratic mean approximation
        return - 0.5 * math.sqrt(Fii * Fjj)
    
        # Tsai-Wu original approximation
        # return -0.5 * Fii

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
    def _compute_F13(cls, failure_stresses: FailureStresses) -> float:
        return cls._compute_Fij(
            cls._compute_F1(failure_stresses),
            cls._compute_F3(failure_stresses),
            cls._compute_F11(failure_stresses),
            cls._compute_F33(failure_stresses),
            failure_stresses,
        )

    @classmethod
    def _compute_F23(cls, failure_stresses: FailureStresses) -> float:
        return cls._compute_Fij(
            cls._compute_F2(failure_stresses),
            cls._compute_F3(failure_stresses),
            cls._compute_F22(failure_stresses),
            cls._compute_F33(failure_stresses),
            failure_stresses,
        )

    @classmethod
    def _compute_A(
        cls, stress_state: StressState, failure_stresses: FailureStresses
    ) -> float:
        if cls._thick_stress_state(stress_state):
            return np.dot(
                np.transpose(stress_state.vector),
                np.dot(cls._F_matrix(failure_stresses), stress_state.vector),
            )[0, 0]
        return (
            cls._compute_F11(failure_stresses) * stress_state.longitudinal**2
            + cls._compute_F22(failure_stresses) * stress_state.traverse**2
            + cls._compute_F66(failure_stresses) * stress_state.shear**2
            - cls._compute_F12(failure_stresses)
            * stress_state.longitudinal
            * stress_state.traverse
        )

    @classmethod
    def _compute_B(
        cls, stress_state: StressState, failure_stresses: FailureStresses
    ) -> float:
        if cls._thick_stress_state(stress_state):
            return np.dot(
                np.transpose(cls._f_vector(failure_stresses)),
                stress_state.vector,
            )[0, 0]
        return (
            cls._compute_F1(failure_stresses) * stress_state.longitudinal
            + cls._compute_F2(failure_stresses) * stress_state.traverse
        )

    @classmethod
    def _F_matrix(cls, failure_stresses: FailureStresses):
        f11 = cls._compute_F11(failure_stresses)
        f22 = cls._compute_F22(failure_stresses)
        f33 = cls._compute_F33(failure_stresses)
        f44 = cls._compute_F44(failure_stresses)
        f55 = cls._compute_F55(failure_stresses)
        f66 = cls._compute_F66(failure_stresses)

        f12 = cls._compute_F12(failure_stresses)
        f13 = cls._compute_F13(failure_stresses)
        f14 = 0
        f15 = 0
        f16 = 0

        f23 = cls._compute_F23(failure_stresses)
        f24 = 0
        f25 = 0
        f26 = 0
        f26 = 0

        f34 = 0
        f35 = 0
        f36 = 0

        f45 = 0
        f46 = 0

        f56 = 0

        return np.array(
            [
                [f11, f12, f13, f14, f15, f16],
                [f12, f22, f23, f24, f25, f26],
                [f13, f23, f33, f34, f35, f36],
                [f14, f24, f34, f44, f45, f46],
                [f15, f25, f35, f45, f55, f56],
                [f16, f26, f36, f46, f56, f66],
            ]
        )

    @classmethod
    def _f_vector(cls, failure_stresses: FailureStresses):
        f1 = cls._compute_F1(failure_stresses)
        f2 = cls._compute_F2(failure_stresses)
        f3 = cls._compute_F3(failure_stresses)
        f4 = 0
        f5 = 0
        f6 = 0

        return np.array([[f1], [f2], [f3], [f4], [f5], [f6]])




def main():
    pass


if __name__ == "__main__":
    main()


# End
