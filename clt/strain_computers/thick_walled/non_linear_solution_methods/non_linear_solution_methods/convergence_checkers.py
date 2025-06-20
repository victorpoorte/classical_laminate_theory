from typing import Protocol
import numpy as np


class ConvergenceChecker(Protocol):
    def has_converged(self, *args: list[float]) -> bool:
        ...


class ForceResidualConvergence(ConvergenceChecker):
    def has_converged(
        self, residual: float, initial_residual: float, tolerance: float
    ) -> bool:
        return np.linalg.norm(residual) <= tolerance * np.linalg.norm(
            initial_residual
        )
    

def main():
    pass


if __name__ == "__main__":
    main()


# End