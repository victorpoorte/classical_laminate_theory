from typing import Protocol
import numpy as np


STRAIN_LABELS = [
    r"$\epsilon_{11}$",
    r"$\epsilon_{22}$",
    r"$\epsilon_{33}$",
    r"$\epsilon_{13}$",
    r"$\epsilon_{23}$",
    r"$\epsilon_{12}$",
]


class Vessel(Protocol):
    ...


class SingleFigure(Protocol):
    def add_data(self, data: np.ndarray) -> None:
        ...

    def create_legend(self, data: np.ndarray) -> None:
        ...


class Line(Protocol):
    ...

class Axis(Protocol):
    ...


class FEData(Protocol):
    def get_through_thickness_strain_vectors(
        self, set_name: str
    ) -> np.ndarray:
        ...


def plot_through_thickness_strains(
    x_values: list[float],
    local_strains: np.ndarray,
    line_initialiser: Line,
    figure_initialiser: SingleFigure,
    x_axis: Axis,
    y_axis: Axis,
) -> SingleFigure:
    strain_lines = [
        line_initialiser(x_values, local_strains[index, :], label)
        for index, label in enumerate(STRAIN_LABELS)
    ]

    figure = figure_initialiser(
        [
            strain_lines[0],
            strain_lines[1],
            strain_lines[2],
            strain_lines[-1],
        ],
        x_axis,
        y_axis,
    )

    return figure


def add_fe_data_to_plot(
    fe_strains: np.ndarray,
    ply_orientations: list[float],
    figure: SingleFigure,
    line_initialiser: Line,
    number_integration_points_per_layer: int = 3,
) -> SingleFigure:
    layer_step = 1 / len(ply_orientations)
    ip_step = layer_step / (number_integration_points_per_layer - 1)
    fe_x_values = [
        i * layer_step + ip_step * j
        for i, _ in enumerate(ply_orientations)
        for j in range(number_integration_points_per_layer)
    ]
    fe_lines = [
        line_initialiser(
            fe_x_values, fe_strains[i, :], label, marker=None, linestyle=""
        )
        for i, label in enumerate(STRAIN_LABELS)
    ]
    figure.add_data(fe_lines[:3] + [fe_lines[-1]])
    figure._create_legend(STRAIN_LABELS[:3] + [STRAIN_LABELS[-1]])

    return figure


def main():
    pass


if __name__ == "__main__":
    main()


# End
