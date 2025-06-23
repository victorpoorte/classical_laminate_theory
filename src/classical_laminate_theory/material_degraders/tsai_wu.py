from ..failure_criteria.failure_criterion_protocol import FailureIndexResult
from ..material import Lamina
from .degrader_protocol import DEGRADED_POISSON, MaterialDegrader, DEGRADATION_FACTOR




class MaterialDegrader(MaterialDegrader):
    degradation_factor = DEGRADATION_FACTOR
    degraded_poisson: float = DEGRADED_POISSON

    def degrade_material(self, lamina: Lamina, failure_index: FailureIndexResult) -> Lamina:

        # Determine which of the failure 
        longitudinal_failure = failure_index.longitudinal >= failure_index.transverse
        
        if longitudinal_failure:
            return self.degrade_longitudinal(lamina)
        return self.degrade_transverse(lamina)

    def degrade_transverse(self, lamina: Lamina) -> Lamina:
        if lamina.degraded_matrix:
            return lamina
        return Lamina(
            lamina.E1,
            self._degrade_modulus(lamina.E2),
            self._degrade_poisson(lamina.v12),
            self._degrade_modulus(lamina.G12),
            E3=lamina.E3,
            v13=self._degrade_poisson(lamina.v13),
            G13=self._degrade_modulus(lamina.G13),
            G23=self._degrade_modulus(lamina.G23),
            v23=self._degrade_poisson(lamina.v23),
            failure_stresses=lamina.failure_stresses,
            degraded_matrix=True,
            alpha1=lamina.alpha1,
            alpha2=lamina.alpha2,
            name=lamina.name,
            p12_negative=lamina.p12_negative,
            p12_positive=lamina.p12_positive,
        )
    
    def degrade_longitudinal(self, lamina: Lamina) -> Lamina:
        if lamina.degraded_fibre:
            return lamina
        return Lamina(
            E1=self._degrade_modulus(lamina.E1),
            E2=self._degrade_modulus(lamina.E2),
            v12=self._degrade_poisson(lamina.v12),
            G12=self._degrade_modulus(lamina.G12),
            E3=self._degrade_modulus(lamina.E3),
            v13=self._degrade_poisson(lamina.v13),
            G13=self._degrade_modulus(lamina.G13),
            G23=self._degrade_modulus(lamina.G23),
            v23=self._degrade_poisson(lamina.v23),
            failure_stresses=lamina.failure_stresses,
            degraded_matrix=True,
            degraded_fibre=True,
            alpha1=lamina.alpha1,
            alpha2=lamina.alpha2,
            name=lamina.name,
            p12_negative=lamina.p12_negative,
            p12_positive=lamina.p12_positive,
        )
    
    def _degrade_modulus(self, shear_modulus: float | None) -> float | None:
        return (
            None
            if shear_modulus is None
            else shear_modulus * self.degradation_factor
        )

    def _degrade_poisson(self, poisson_ratio: float | None) -> float | None:
        return None if poisson_ratio is None else self.degraded_poisson


def main():
    pass


if __name__ == "__main__":
    main()


# End
