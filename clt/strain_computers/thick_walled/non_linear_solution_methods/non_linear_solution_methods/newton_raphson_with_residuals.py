from dataclasses import dataclass
from typing import Protocol
import numpy as np

from .solution_method import NonLinearSolutionMethodWithSoftening


class ConvergenceChecker(Protocol):
    def has_converged(self, *args: list[float]) -> bool:
        ...


@dataclass
class NewtonRaphsonWithResiduals(NonLinearSolutionMethodWithSoftening):
    convergence_checkers: list[ConvergenceChecker]
    max_iterations: int = int(1e3)
    tolerance: float = 1e-3

    def compute_solutions(
        self,
        internal_load: callable,
        internal_load_derivative: callable,
        external_load: callable,
        external_load_derivative: callable,
        x0: np.ndarray,
    ) -> np.ndarray:
        a = x0.copy()
        for iteration in range(int(self.max_iterations)):
            stiffness_matrix: np.ndarray = internal_load_derivative(
                a
            ) - external_load_derivative(a)
            if not stiffness_matrix.any():
                print("Derivative in null... Failed with Newton-Raphson.")
                return float("nan")

            residual = external_load(a) - internal_load(a)
            if iteration == 0:
                initial_residual = residual
            if self.converged(residual, initial_residual, self.tolerance):
                return a

            delta_a = np.dot(np.linalg.inv(stiffness_matrix), residual)
            a += delta_a

        raise StopIteration("Convergence failure...")


def main():
    pass


if __name__ == "__main__":
    main()


# End
