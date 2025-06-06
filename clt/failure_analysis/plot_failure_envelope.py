import numpy as np
from clt.clt.failure_analysis.failure_envelope import FailureEnvelope
from plotting import SingleFigure, SingleLine, Axis


def plot_lpf_failure_envelope(
    envelope: FailureEnvelope,
    x_label: str,
    y_label: str,
) -> SingleFigure:
    first = SingleLine(
        envelope.first_x_values, envelope.first_y_values, label="FPF"
    )
    last = SingleLine(
        envelope.last_x_values, envelope.last_y_values, label="FPF"
    )

    x_axis = Axis(None, None, None, x_label)
    y_axis = Axis(None, None, None, y_label)
    fig = SingleFigure([first, last], x_axis, y_axis)

    return fig

def plot_fpf_failure_envelope(
    envelopes: list[FailureEnvelope],
    labels: list[str],
    x_label: str,
    y_label: str,
) -> SingleFigure:

    lines = [
        SingleLine(envelope.first_x_values, envelope.first_y_values, label=label)
        for label, envelope in zip(labels, envelopes)
    ]

    x_axis = Axis(None, None, None, x_label)
    y_axis = Axis(None, None, None, y_label)
    fig = SingleFigure(lines, x_axis, y_axis)

    return fig


# End
