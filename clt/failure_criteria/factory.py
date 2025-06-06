from .failure_criterion_protocol import FailureCriterion
from . import hashin, tsai_wu, puck, tsai_wu_thick, maximum_stress

class FailureCriteriaFactory:
    _failure_criteria = {
        "Tsai-Wu": tsai_wu.FailureCriterion(),
        "Tsai-Wu Thick": tsai_wu_thick.FailureCriterion(),
        "Hashin": hashin.FailureCriterion(),
        "Puck": puck.FailureCriterion(),
        "Max Stress": maximum_stress.FailureCriterion()
    }

    @property
    def available(self):
        return ", ".join(self._failure_criteria.keys())

    def get_failure_criterion(self, criterion: str) -> FailureCriterion:
        crit = self._failure_criteria.get(criterion)
        if crit is None:
            raise ValueError(
                f"\n{criterion} is not available as failure criterion.\n"
                + f"Available criteria are: {self.available}.\n"
            )
        return crit