from dataclasses import dataclass
import numpy as np

from .failure_criterion_protocol import FailureCriterion, FailureMode, Layer, classify_failure_mode, FailureIndexResult

class FailureCriterion(FailureCriterion):
    name = "Puck"

    def factor_of_safety(self, global_strain: np.ndarray, layer: Layer, temperature_delta: float, operating_temperature: float) -> float:

        failure_index_result  = self.long_and_trav_index(
            global_strain, layer, temperature_delta, operating_temperature
        )

        fibre_index = failure_index_result.longitudinal
        matrix_index = failure_index_result.transverse
        matrix_failure_angle = failure_index_result.failure_angle
        fibre_tensile_failure = failure_index_result.long_tensile
        matrix_tension_failure = failure_index_result.trav_tensile

        # Convert indices to factors
        fibre_factor = 1 / fibre_index
        matrix_factor = 1 / matrix_index
        
        indices = np.array([fibre_index, matrix_index])
        failure_index = max(indices)
        factor_of_safety = 1 / failure_index

        
        failure_type = classify_failure_mode(fibre_factor, fibre_tensile_failure, matrix_factor, matrix_tension_failure)
        failure_mode = FailureMode(self.name, failure_type, matrix_failure_angle)

        return factor_of_safety, failure_mode
    
    def long_and_trav_index(self, global_strain: np.ndarray, layer: Layer, temperature_delta: float, operating_temperature: float) -> FailureIndexResult:
        layer_stress = self._compute_layer_stress_state(layer, global_strain, temperature_delta)

        # Unpack stress vector
        stress_vector = layer_stress.vector.flatten()
        sigma_1 = stress_vector[0]
        sigma_2 = stress_vector[1]
        sigma_3 = stress_vector[2]
        tau_23 = stress_vector[3]
        tau_13 = stress_vector[4]
        tau_21 = stress_vector[5]

        lamina_failure_stresses = layer.lamina.get_failure_stress(operating_temperature)
        X_t = lamina_failure_stresses.X_t
        X_c = lamina_failure_stresses.X_c
        Y_t = lamina_failure_stresses.Y_t
        Y_c = lamina_failure_stresses.Y_c
        S = lamina_failure_stresses.S

        fibre_index, fibre_tensile_failure, matrix_index, matrix_tension_failure, matrix_failure_angle = self._failure_indices_for_stress(
            sigma_1, sigma_2, sigma_3, tau_21, tau_23, tau_13, layer.lamina.p12_negative, layer.lamina.p12_positive, X_t, X_c, Y_t, Y_c, S
        )

        return FailureIndexResult(fibre_index, matrix_index, matrix_failure_angle, fibre_tensile_failure, matrix_tension_failure)

    def _failure_indices_for_stress(
        self, sigma_1, sigma_2, sigma_3, tau_21, tau_23, tau_31, p_12_negative, p_12_positive, X_t, X_c, Y_t, Y_c, S
    ):
        
        # Convert variables to Puck names
        R_np = S
        R_n = Y_t
        Y_c = abs(Y_c)

        # Compute necessary values
        p_22_negative = compute_p_nn_negative(p_12_negative, R_np, Y_c)
        R_nn = compute_R_nn(Y_c, p_22_negative)
        p_22_positive = compute_p_22_positive(p_12_positive, R_np, R_nn)
        
        fibre_factor = self._fibre_failure_index(sigma_1, X_t, X_c)
        matrix_factor, matrix_failure_angle, matrix_tension_failure = self._inter_fibre_failure_index(
            sigma_2, sigma_3, tau_21, tau_23, tau_31, R_n, R_np, R_nn, p_12_negative, p_12_positive, p_22_negative, p_22_positive,
        )
        
        fibre_tensile_failure = sigma_1 >= 0

        return fibre_factor, fibre_tensile_failure, matrix_factor, matrix_tension_failure, matrix_failure_angle

    def _inter_fibre_failure_index(self, sigma_2: float, sigma_3: float, tau_21: float, tau_23: float, tau_31: float, R_n: float, R_np: float, R_nn: float, p_12_negative: float, p_12_positive: float, p_22_negative: float, p_22_positive: float) -> float:

        # Define range of failure planes
        failure_angles = np.radians(np.arange(-90, 91, 1))

        # Compute stresses for failure planes
        sigma_n = compute_sigma_n(sigma_2, sigma_3,tau_23, failure_angles)

        # Compute tau terms
        tau_nt = compute_tau_nt(sigma_2, sigma_3, tau_23, failure_angles)
        tau_n1 = compute_tau_n1(tau_21, tau_31, failure_angles)

        # Compute psi terms
        cos2_psi = compute_cos2_psi(tau_nt, tau_n1)
        sin2_psi = compute_sin2_psi(tau_nt, tau_n1)

        # Mask used to select the appropriate compression and tension cases
        sigma_positive = sigma_n >= 0

        # Create lists for positive and negative sigma_n values
        p_22 = np.where(sigma_positive, p_22_positive, p_22_negative)
        p_12 = np.where(sigma_positive, p_12_positive, p_12_negative)

        # Compute p/R__psi ratios
        p_R_psi_ratio = compute_p_R_psi_ratio(p_22, p_12, R_nn, R_np, cos2_psi, sin2_psi)

        # Compute failure indices for crack angles
        args = [sigma_n, tau_nt, tau_n1, R_n, R_nn, R_np, p_R_psi_ratio]
        f_E_thetas = np.where(
            sigma_positive,
            compute_f_E_theta_tension(*args),
            compute_f_E_theta_compression(*args)
        )

        # Extract failure angle, critical index, if tensile failure
        failure_index = np.argmax(f_E_thetas)
        failure_angle = np.degrees(failure_angles[failure_index])
        f_E = f_E_thetas[failure_index]
        tensile_failure = sigma_positive[failure_index]

        return f_E, failure_angle, tensile_failure
    
    def _fibre_failure_index(self, sigma_1: float, X_t: float, X_c: float) -> float:
        return sigma_1 / (X_t if sigma_1 >= 0 else X_c)

def compute_p_22_positive(p_12_positive, R_np, R_nn):
    p_22_positive = p_12_positive * R_nn / R_np
    return p_22_positive

def compute_R_nn(Y_c, p_22_negative):
    R_nn = Y_c / (2 * (1 + p_22_negative))
    return R_nn

def compute_p_nn_negative(p_12_negative, R_np, Y_c):
    a = 1
    b = 1
    c = - (p_12_negative * Y_c) / (R_np * 2)
    p_22_negative = (- b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    return p_22_negative

def compute_sigma_n(sigma_2: float, sigma_3: float, tau_23: float, theta: float) -> float:
    cos = np.cos(theta)
    sin = np.sin(theta)
    
    return (
        sigma_2 * cos ** 2
        + sigma_3 * sin ** 2
        + 2 * tau_23 * sin * cos
    )

def compute_tau_nt(sigma_2: float, sigma_3: float, tau_23: float, theta: float) -> float:
    cos = np.cos(theta)
    sin = np.sin(theta)

    return (
        (sigma_3 - sigma_2) * sin * cos 
        + tau_23 * (cos ** 2 - sin ** 2)
    )

def compute_tau_n1(tau_21: float, tau_31, theta: float) -> float:
    cos = np.cos(theta)
    sin = np.sin(theta)

    return tau_31 * sin + tau_21 * cos

def compute_f_E_theta_tension(sigma_n: float, tau_nt: float, tau_n1: float, R_n: float, R_nn: float, R_np: float, p_R_psi_ratio: float) -> float:
    return (
        np.sqrt(
            ((1 / R_n - p_R_psi_ratio) * sigma_n) ** 2
            + (tau_nt / R_nn) ** 2
            + (tau_n1 / R_np) ** 2
        )
        + p_R_psi_ratio * sigma_n
    )

def compute_f_E_theta_compression(sigma_n: float, tau_nt: float, tau_n1: float, R_n: float, R_nn: float, R_np: float, p_R_psi_ratio: float) -> float:
    return (
        np.sqrt(
            (tau_nt / R_nn) ** 2
            + (tau_n1 / R_np) ** 2
            + (p_R_psi_ratio * sigma_n) ** 2
        )
        + p_R_psi_ratio * sigma_n
    )

def compute_cos2_psi(tau_nt: float, tau_n1: float) -> float:
    num = tau_nt ** 2
    den = (tau_nt ** 2 + tau_n1 ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos2_psi = np.where(num == 0, 0, num / den)
    return cos2_psi

def compute_sin2_psi(tau_nt: float, tau_n1: float) -> float:
    num = tau_n1 ** 2
    den = (tau_nt ** 2 + tau_n1 ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        sin2_psi = np.where(num == 0, 0, num / den)
    return sin2_psi

def compute_p_R_psi_ratio(p_nn: float, p_np: float, R_nn: float, R_np: float, cos2_psi: float, sin2_psi: float) -> float:
    return p_nn / R_nn * cos2_psi + p_np / R_np * sin2_psi

def compute_p_np(p_nn: float, R_nn: float, R_np: float) -> float:

    return p_nn * R_np / R_nn


# End