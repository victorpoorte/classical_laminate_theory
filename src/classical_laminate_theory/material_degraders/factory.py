from .tsai_wu import MaterialDegrader as TsaiWu
from .degrader_protocol import MaterialDegrader

class MaterialDegraderFactory:
    degraders = {
        "Tsai-Wu": TsaiWu()
    }

    @property
    def available(self):
        return ", ".join(self.degraders.keys())

    def create(self, degrader: str) -> MaterialDegrader:
        deg = self.degraders.get(degrader)

        if deg is not None:
            return deg

        raise ValueError(
            f"{degrader} not available as material degrader.\n"
            + f"Available degraders are: {self.available}"
        )
