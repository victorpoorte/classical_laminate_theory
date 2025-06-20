import math
from abc import abstractmethod
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ..tube_layers import TubeLayer


@dataclass
class TubeLayer(TubeLayer):
    thickness: float
    C11: float
    C22: float
    C33: float
    C12: float
    C13: float
    C23: float
    C16: float
    C26: float
    C36: float
    C66: float
    xi: np.ndarray

    @property
    def xi_r(self):
        return self.xi[2, 0]

    @property
    def xi_theta(self):
        return self.xi[1, 0]

    @property
    def xi_z(self):
        return self.xi[0, 0]

    @property
    def xi_z_theta(self):
        return self.xi[-1, 0]

    @property
    def eta(self):
        return (self.xi_r - self.xi_theta) / self.C33

    @property
    def beta(self):
        raise NotImplementedError("Beta should be implemented in subclasses")

    @cached_property
    def alpha_2(self):
        return (self.C26 - 2 * self.C36) / (4 * self.C33 - self.C22)

    def first_stress_entry(self, radius: float) -> float:
        return (self.C23 + self.beta * self.C33) * radius ** (self.beta - 1)

    def second_stress_entry(self, radius: float) -> float:
        return (self.C23 - self.beta * self.C33) * radius ** (-self.beta - 1)

    def twist_stress_entry(self, radius: float) -> float:
        return ((self.C36) + self.alpha_2 * (self.C23 + 2 * self.C33)) * radius

    def first_displacement_entry(self, radius: float) -> float:
        return radius**self.beta

    def second_displacement_entry(self, radius: float) -> float:
        return radius ** (-self.beta)

    def twist_displacement_entry(self, radius: float) -> float:
        return self.alpha_2 * radius**2

    def first_elongation_entry(
        self, radius_1: float, radius_2: float
    ) -> float:
        factor_1 = (self.C12 + self.beta * self.C13) / (1 + self.beta)
        factor_2 = radius_2 ** (self.beta + 1) - radius_1 ** (self.beta + 1)
        return factor_1 * factor_2

    def twist_elongation_entry(
        self, radius_1: float, radius_2: float
    ) -> float:
        return (
            (self.C16 + self.alpha_2 * (self.C12 + 2 * self.C13))
            * (radius_2**3 - radius_1**3)
            / 3
        )

    def first_twist_entry(self, radius_1: float, radius_2: float) -> float:
        numerator = self.C26 + self.beta * self.C36
        denominator = 2 + self.beta
        factor_2 = radius_2 ** (self.beta + 2) - radius_1 ** (self.beta + 2)

        return numerator / denominator * factor_2

    def second_twist_entry(self, radius_1: float, radius_2: float) -> float:
        numerator = self.C26 - self.beta * self.C36
        denominator = 2 - self.beta
        factor_2 = radius_2 ** (-self.beta + 2) - radius_1 ** (-self.beta + 2)

        return numerator / denominator * factor_2

    def twist_twist_entry(self, radius_1: float, radius_2: float) -> float:
        return (
            (self.C66 + self.alpha_2 * (self.C26 + 2 * self.C36))
            * (radius_2**4 - radius_1**4)
            / 4
        )

    def d_epsilon_d_D(self, radius: float) -> float:
        return radius ** (self.beta - 1)

    def d_epsilon_d_E(self, radius: float) -> float:
        return radius ** (-self.beta - 1)

    def d_epsilon_theta_d_solution(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
    ) -> np.ndarray:
        array = np.zeros(solution_vector.size)
        array[layer_index] = self.d_epsilon_d_D(radius)
        array[layer_index + number_of_layers] = self.d_epsilon_d_E(radius)
        array[-2] = self.d_epsilon_d_epsilon_0(radius)
        array[-1] = self.d_epsilon_d_gamma_0(radius)
        return array

    @classmethod
    def _unpack_solution_vector(
        cls,
        solution_vector: np.ndarray,
        layer_index: int,
    ) -> tuple[float, float, float, float]:
        number_of_layers = cls.number_of_layers(solution_vector)
        d = cls._unpack_D_from_solution_vector(
            solution_vector,
            layer_index,
            number_of_layers,
        )
        e = cls._unpack_E_from_solution_vector(
            solution_vector, layer_index, number_of_layers
        )
        epsilon_0 = cls._unpack_epsilon_from_solution_vector(
            solution_vector, layer_index, number_of_layers
        )
        gamma_0 = cls._unpack_gamma_from_solution_vector(
            solution_vector, layer_index, number_of_layers
        )

        return d, e, epsilon_0, gamma_0

    @staticmethod
    def _unpack_D_from_solution_vector(
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
    ) -> float:
        return solution_vector[layer_index, 0]

    @staticmethod
    def _unpack_E_from_solution_vector(
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
    ) -> float:
        return solution_vector[layer_index + number_of_layers, 0]

    @staticmethod
    def _unpack_epsilon_from_solution_vector(
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
    ) -> float:
        return solution_vector[-2, 0]

    @staticmethod
    def _unpack_gamma_from_solution_vector(
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
    ) -> float:
        return solution_vector[-1, 0]

    @staticmethod
    def number_of_layers(solution_vector: np.ndarray) -> int:
        return int((len(solution_vector) - 2) / 2)

    # @abstractmethod
    def d_epsilon_d_epsilon_0(self, radius: float) -> float:
        ...

    def d_epsilon_d_gamma_0(self, radius: float) -> float:
        return self.alpha_2 * radius

    # @abstractmethod
    def load_derivative_displacement_entry(
        self,
        interface_radius: float,
        epsilon_theta: float,
        d_epsilon_d_solution: np.ndarray,
    ) -> float:
        ...

    # @abstractmethod
    def load_derivative_elongation_entry(
        self,
        radius: float,
        epsilon_theta: float,
        d_epsilon_d_solution: np.ndarray,
    ) -> np.ndarray:
        ...

    # @abstractmethod
    def load_derivative_twist_entry(
        self,
        radius: float,
        epsilon_theta: float,
        d_epsilon_d_solution: np.ndarray,
    ) -> np.ndarray:
        ...


class Case1Layer(TubeLayer):
    # Case when beta is > 0

    @cached_property
    def alpha_1(self):
        num = self.C12 - self.C13
        if num == 0:
            return 0
        return num / (self.C33 - self.C22)

    @cached_property
    def beta(self):
        return math.sqrt(self.C22 / self.C33)

    @property
    def load_factor(self) -> float:
        return self.eta / (1 - self.beta**2)

    def elongation_stress_entry(self, _: float) -> float:
        return self.C13 + self.alpha_1 * (self.C23 + self.C33)

    def elongation_displacement_entry(self, radius: float) -> float:
        return self.alpha_1 * radius

    def second_elongation_entry(
        self, radius_1: float, radius_2: float
    ) -> float:
        if self.beta == 1:
            return 0
        factor_1 = (self.C12 - self.beta * self.C13) / (1 - self.beta)
        factor_2 = radius_2 ** (-self.beta + 1) - radius_1 ** (-self.beta + 1)
        return factor_1 * factor_2

    def elongation_elongation_entry(
        self, radius_1: float, radius_2: float
    ) -> float:
        return (self.C11 + self.alpha_1 * (self.C12 + self.C13)) * (
            (radius_2**2 - radius_1**2) / 2
        )

    def elongation_twist_entry(
        self, radius_1: float, radius_2: float
    ) -> float:
        return (
            (self.C16 + self.alpha_1 * (self.C26 + self.C36))
            * (radius_2**3 - radius_1**3)
            / 3
        )

    @property
    def load_vector_stress_factor(self):
        return self.load_factor * (self.C33 + self.C23) - self.xi_r

    def load_vector_stress_entry(self, radius: float) -> float:
        return self.load_vector_stress_factor

    def load_vector_displacement_entry(self, radius: float) -> float:
        return self.load_factor * radius

    @property
    def load_vector_elongation_factor(self):
        return self.load_factor * (self.C13 + self.C12) - self.xi_z

    def load_vector_elongation_entry(
        self, radius1: float, radius2: float
    ) -> float:
        return self.load_vector_elongation_factor * (
            radius2**2 - radius1**2
        )

    @property
    def load_vector_twist_factor(self):
        return (self.C26 + self.C36) * self.eta / (1 - self.beta**2) / 3

    def load_vector_twist_entry(self, radius1: float, radius2: float) -> float:
        return self.load_vector_twist_factor * (radius2**3 - radius1**3)

    def u(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
        temperature_delta: float,
    ) -> float:
        d, e, epsilon_0, gamma_0 = self._unpack_solution_vector(
            solution_vector, layer_index
        )
        return (
            d * radius**self.beta
            + e * radius ** (-self.beta)
            + self.alpha_1 * epsilon_0 * radius
            + self.alpha_2 * gamma_0 * radius**2
            + self.eta * temperature_delta / (1 - self.beta**2) * radius
        )

    def du_dr(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
        temperature_delta: float,
    ) -> float:
        d, e, epsilon_0, gamma_0 = self._unpack_solution_vector(
            solution_vector, layer_index
        )
        return (
            d * self.beta * radius ** (self.beta - 1)
            - e * self.beta * radius ** (-self.beta - 1)
            + self.alpha_1 * epsilon_0
            + 2 * self.alpha_2 * gamma_0 * radius
            + self.eta * temperature_delta / (1 - self.beta**2)
        )

    def d_epsilon_d_epsilon_0(self, radius: float) -> float:
        return self.alpha_1

    def load_derivative_displacement_entry(
        self,
        interface_radius: float,
        epsilon_theta: float,
        d_epsilon_d_solution: np.ndarray,
    ) -> float:
        return self.load_factor * interface_radius * d_epsilon_d_solution

    def load_derivative_elongation_entry(
        self,
        radius: float,
        epsilon_theta: float,
        d_epsilon_d_solution: np.ndarray,
    ) -> np.ndarray:
        return (
            2
            * self.load_vector_elongation_factor
            * (1 + epsilon_theta)
            * d_epsilon_d_solution
            * radius**2
        )

    def load_derivative_twist_entry(
        self,
        radius: float,
        epsilon_theta: float,
        d_epsilon_d_solution: np.ndarray,
    ) -> np.ndarray:
        return (
            3
            * self.load_vector_twist_factor
            * (1 + epsilon_theta)
            * d_epsilon_d_solution
            * radius**3
        )


class Case2Layer(TubeLayer):
    # Case when beta is one

    @property
    def beta(self):
        return 1

    @property
    def alpha_3(self):
        return (self.C12 - self.C13) / self.C33

    def compliance_factor(self, radius: float):
        return self.alpha_3 * math.log(radius) / 2

    def elongation_stress_entry(self, radius: float):
        return 0.5 * (
            self.alpha_3 * math.log(radius) * (self.C33 + self.C23)
            + self.alpha_3 * self.C33
            + 2 * self.C13
        )

    def second_elongation_entry(self, radius1: float, radius2: float) -> float:
        return 2 * (self.C12 - self.C13) * math.log(radius2 / radius1)

    def elongation_elongation_entry(
        self, radius1: float, radius2: float
    ) -> float:
        return (
            self.alpha_3
            * (self.C13 + self.C12)
            * (
                (
                    radius2**2 * math.log(radius2)
                    - radius1**2 * math.log(radius1)
                )
                / 4
                - (radius2**2 - radius1**2) / 8
            )
            + (self.alpha_3 * self.C13 / 2 + self.C11)
            * (radius2**2 - radius1**2)
            / 2
        )

    def load_vector_stress_entry(self, radius: float):
        return (
            self.eta * math.log(radius) * (self.C33 + self.C23)
            + self.eta * self.C33
        ) / 2 - self.xi_r

    def load_vector_displacement_entry(self, radius: float) -> float:
        return self.eta * radius / 2 * math.log(radius)

    def load_vector_elongation_entry(self, radius1: float, radius2: float):
        return (
            self.eta
            / 2
            * (self.C13 + self.C12)
            * (
                (
                    radius2**2 * math.log(radius2)
                    - radius1**2 * math.log(radius1)
                )
                / 2
                - (radius2**2 - radius1**2) / 4
            )
            + (self.eta / 2 * self.C13 - self.xi_z)
            * (radius2**2 - radius1**2)
            / 2
        )

    def elongation_displacement_entry(self, radius: float) -> float:
        return self.alpha_3 * radius * math.log(radius) / 2

    def elongation_twist_integral_entry(self, radius: float) -> float:
        return (
            (
                self.C16
                - self.alpha_3 * self.C26 / 6
                + self.alpha_3 * self.C36 / 3
                + self.alpha_3 * (self.C26 + self.C36) * math.log(radius) / 2
            )
            * (radius**3)
            / 3
        )

    def elongation_twist_entry(
        self, radius_1: float, radius_2: float
    ) -> float:
        return self.elongation_twist_integral_entry(
            radius_2
        ) - self.elongation_twist_integral_entry(radius_1)

    def load_twist_integral_entry(self, radius: float) -> float:
        return (
            self.eta
            / 6
            * radius**3
            * (
                3 * math.log(radius) * (self.C26 + self.C36)
                - self.C26
                + 2 * self.C36
                - 2 * self.xi_z_theta
            )
        )

    def load_vector_twist_entry(self, radius1: float, radius2: float) -> float:
        return self.load_twist_integral_entry(
            radius2
        ) - self.load_twist_integral_entry(radius1)

    def u(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
        temperature_delta: float,
    ) -> float:
        a, b, epsilon_0, gamma0 = self._unpack_solution_vector(
            solution_vector, layer_index
        )
        return (
            (self.alpha_3 * epsilon_0 + self.eta * temperature_delta)
            * radius
            * math.log(radius)
            / 2
            + a * radius
            + b / radius
            + self.alpha_2 * gamma0 * radius**2
        )

    def du_dr(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
        temperature_delta: float,
    ) -> float:
        a, b, epsilon_0, gamma0 = self._unpack_solution_vector(
            solution_vector, layer_index
        )
        return (
            (self.alpha_3 * epsilon_0 + self.eta * temperature_delta)
            * (math.log(radius) + 1)
            / 2
            + a
            - b / (radius**2)
            + 2 * self.alpha_2 * gamma0 * radius
        )

    def d_epsilon_d_epsilon_0(self, radius: float) -> float:
        return self.alpha_3 / 2 * math.log(radius)

    def load_derivative_displacement_entry(
        self,
        interface_radius: float,
        epsilon_theta: float,
        d_epsilon_d_solution: np.ndarray,
    ) -> float:
        new_radius = compute_new_radius(interface_radius, epsilon_theta)
        d_r_solution = d_radius_d_solution(
            interface_radius, d_epsilon_d_solution
        )
        return self.eta / 2 * d_r_solution * (math.log(new_radius) + 1)

    def load_derivative_elongation_entry(
        self,
        radius: float,
        epsilon_theta: float,
        d_epsilon_d_solution: np.ndarray,
    ) -> np.ndarray:
        new_radius = compute_new_radius(radius, epsilon_theta)
        d_r_d_solution = d_radius_d_solution(radius, d_epsilon_d_solution)
        return (
            self.eta
            * (self.C13 + self.C12)
            * (
                new_radius * d_r_d_solution * (2 * math.log(new_radius) + 1)
                - 2 * new_radius * d_r_d_solution
            )
            + (self.eta * self.C13 / 2 - self.xi_z)
            * 2
            * new_radius
            * d_r_d_solution
        )

    def load_derivative_twist_entry(
        self,
        radius: float,
        epsilon_theta: float,
        d_epsilon_d_solution: np.ndarray,
    ) -> np.ndarray:
        new_radius = compute_new_radius(radius, epsilon_theta)
        d_r_d_solution = d_radius_d_solution(radius, d_epsilon_d_solution)
        return (
            self.eta
            / 2
            * (
                new_radius**2
                * d_r_d_solution
                * (math.log(new_radius) + 1 / 3)
                * (self.C26 + self.C36)
                + new_radius**2
                / 3
                * d_r_d_solution
                * (2 * self.C36 - self.C26)
            )
            - self.xi_z_theta * new_radius**2 * d_r_d_solution
        )


class Case3Layer(TubeLayer):

    # Case when beta is < 0

    @cached_property
    def beta(self):
        return math.sqrt(- self.C22 / self.C33)

    # @cached_property
    # def alpha(self):
    #     return (self.C12 - self.C13) / self.C33
    
    @cached_property
    def alpha_1(self):
        num = self.C12 - self.C13
        if num == 0:
            return 0
        return num / (self.C33 - self.C22)

    @cached_property
    def alpha_2(self):
        return (self.C26 - 2 * self.C36) / (4 * self.C33 - self.C22)

    # def compliance_factor(self, radius: float):
    #     return self.alpha * math.log(radius) / 2

    # @property
    # def load_factor(self, radius: float):
    #     return self.eta * math.log(radius) / 2

    # def compliance_factor(self, _: float):
    #     return self.alpha / (1 + self.beta**2)

    # def load_factor(self, _: float):
    #     return self.eta / (1 + self.beta**2)

    def first_stress_entry(self, radius: float) -> float:
        return (
            -self.beta * self.C33 * math.sin(self.beta * math.log(radius))
            + self.C23 * math.cos(self.beta * math.log(radius))
        ) / radius

    def second_stress_entry(self, radius: float) -> float:
        return (
            self.beta * self.C33 * math.cos(self.beta * math.log(radius))
            + self.C23 * math.sin(self.beta * math.log(radius))
        ) / radius

    def elongation_stress_entry(self, _: float) -> float:
        return (self.C23 + self.C33)*self.alpha_1 + self.C13

    def twist_stress_entry(self, radius: float) -> float:
        return ((self.C23 + 2 * self.C33) * self.alpha_2 + self.C36) * radius

    def first_displacement_entry(self, radius: float) -> float:
        return math.cos(self.beta * math.log(radius))

    def second_displacement_entry(self, radius: float) -> float:
        return math.sin(self.beta * math.log(radius))

    def elongation_displacement_entry(self, radius: float) -> float:
        return self.alpha_1 * radius

    def twist_displacement_entry(self, radius: float) -> float:
        return self.alpha_2 * radius ** 2
    
    def cos(self, radius: float) -> float:
        return np.cos(self.beta * np.log(radius))

    def sin(self, radius: float) -> float:
        return np.sin(self.beta * np.log(radius))

    def first_elongation_entry(self, radius1: float, radius2: float) -> float:
        def entry(radius):
            return (
                radius
                * (
                    (self.C13 * self.beta**2 + self.C12) * self.cos(radius)
                    + (self.C12 - self.C13) * self.beta * self.sin(radius)
                )
                / (1 + self.beta**2)
            )
        return entry(radius2) - entry(radius1)

    def second_elongation_entry(self, radius1: float, radius2: float) -> float:
        def entry(radius):
            return (
                radius
                * (
                    (self.C13 * self.beta**2 + self.C12) * self.sin(radius)
                    - (self.C12 - self.C13) * self.beta * self.cos(radius)
                )
                / (1 + self.beta**2)
            )
        return entry(radius2) - entry(radius1)

    def elongation_elongation_entry(self, radius1: float, radius2: float) -> float:
        def entry(radius):
            return (((self.C12 + self.C13) * self.alpha_1 + self.C11) * radius**2) / 2
        
        return entry(radius2) - entry(radius1)

    def twist_elongation_entry(self, radius1: float, radius2: float) -> float:
        def entry(radius):
            return  (((self.C12 + 2*self.C13)*self.alpha_2 + self.C16)*radius**3)/3
        
        return entry(radius2) - entry(radius1)

    def first_twist_entry(self, radius1: float, radius2: float) -> float:
        def entry(radius):
            return (
                (
                    (self.C36*self.beta**2 + 2*self.C26) * self.cos(radius)
                    + (self.C26 - 2*self.C36) * self.beta * self.sin(radius)
                ) *radius**2
                / (self.beta**2 + 4)
            )
        
        return entry(radius2) - entry(radius1)

    def second_twist_entry(self, radius1: float, radius2: float) -> float:
        def entry(radius):
            return (
                (
                    (self.C36*self.beta**2 + 2*self.C26) * self.sin(radius)
                    - (self.C26 - 2*self.C36) * self.beta * self.cos(radius)
                ) *radius**2
                / (self.beta**2 + 4)
            )
        
        return entry(radius2) - entry(radius1)
    
    def elongation_twist_entry(self, radius1: float, radius2: float) -> float:
        def entry(radius):
            return (((self.C26 + self.C36)*self.alpha_1 + self.C16)*radius**3)/3
        
        return entry(radius2) - entry(radius1)

    def twist_twist_entry(self, radius1: float, radius2: float) -> float:
        def entry(radius):
            return (((self.C26 + 2*self.C36)*self.alpha_2 + self.C66)*radius**4)/4
        
        return entry(radius2) - entry(radius1)

    def u(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
        temperature_delta: float,
    ) -> float:
        a, b, epsilon_0, gamma_0 = self._unpack_solution_vector(
            solution_vector, layer_index
        )
        return (
            a * math.cos(self.beta * math.log(radius))
            + b * math.sin(self.beta * math.log(radius))
            + self.alpha_1 * epsilon_0 * radius
            + self.alpha_2 * gamma_0 * radius ** 2
            + (self.eta * temperature_delta * radius / (1 - self.beta**2))
        )

    def du_dr(
        self,
        radius: float,
        solution_vector: np.ndarray,
        layer_index: int,
        number_of_layers: int,
        temperature_delta: float,
    ) -> float:
        a, b, epsilon_0, gamma_0 = self._unpack_solution_vector(
            solution_vector, layer_index
        )
        return (
            -a * self.beta * math.sin(self.beta * math.log(radius)) / radius
            + b * self.beta * math.cos(self.beta * math.log(radius)) / radius
            + self.alpha_1 * epsilon_0
            + 2 * self.alpha_2 * gamma_0 * radius
            + (self.eta * temperature_delta / (1 - self.beta**2))
        )

    def load_vector_displacement_entry(self, radius: float) -> float:
        return - self.eta*radius / (1 - self.beta**2)

    def load_vector_stress_entry(self, radius: float) -> float:
        return (self.C23 + self.C33) * self.eta / (self.beta**2 - 1)

    def load_vector_elongation_entry(self, radius1: float, radius2: float) -> float:
        def entry(radius):
            return (self.C12 + self.C13)*self.eta*radius**2/(2*self.beta**2 - 2)
        
        return entry(radius2) - entry(radius1)

    def load_vector_twist_entry(self, radius1: float, radius2: float) -> float:
        def entry(radius):
            return self.eta*radius**3*(self.C26 + self.C36)/(3*self.beta**2 - 3)
        
        return entry(radius2) - entry(radius1)

    

def d_radius_d_solution(
    radius: float, d_epsilon_d_solution: np.ndarray
) -> np.ndarray:
    return radius * d_epsilon_d_solution


def compute_new_radius(original_radius: float, epsilon_theta: float) -> float:
    """Function to compute the new radius given a strain in the tangent
    directions.

    Args:
        original_radius (float): Original radius of the vessel.
        epsilon_theta (float): Strain in the theta/tangent direction.

    Returns:
        float: New radius as consequence to applied strain.
    """
    return original_radius * (1 + epsilon_theta)


def main():
    pass


if __name__ == "__main__":
    main()


# End
