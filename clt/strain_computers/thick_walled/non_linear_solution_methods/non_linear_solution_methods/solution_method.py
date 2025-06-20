from abc import abstractmethod
import numpy as np
from typing import Protocol


class ConvergenceChecker(Protocol):
    def has_converged(self, *args: list[float]) -> bool:
        ...


class NonLinearSolutionMethod(Protocol):
    convergence_checkers: list[ConvergenceChecker]
    max_iterations: int
    tolerance: float

    @abstractmethod
    def compute_solutions(
        self,
        internal_load: callable,
        external_load: callable,
        x0: np.ndarray,
    ) -> np.ndarray:
        ...

    def converged(self, *args: list[float]) -> bool:
        for checker in self.convergence_checkers:
            if not checker.has_converged(*args):
                return False
        return True


class NonLinearSolutionMethodWithSoftening(NonLinearSolutionMethod):
    @abstractmethod
    def compute_solutions(
        self,
        internal_load: callable,
        internal_load_derivative: callable,
        external_load: callable,
        external_load_derivative: callable,
        x0: np.ndarray,
    ) -> np.ndarray:
        ...


def main():
    pass


if __name__ == "__main__":
    main()


# End
