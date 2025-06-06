import hypothesis.strategies as st


@st.composite
def my_positive_number_strategy(draw):
    return float(
        draw(
            st.decimals(allow_nan=False, allow_infinity=False, min_value=1e-10)
        )
    )


@st.composite
def my_list_of_numbers_strategy(
    draw, no_numbers: float, min_value=0.1
) -> list[float]:
    kwargs = {"min_value": min_value}
    if min_value is False:
        kwargs = dict()
    vars = draw(
        st.lists(
            st.decimals(allow_nan=False, allow_infinity=False, **kwargs),
            min_size=no_numbers,
            max_size=no_numbers,
        )
    )

    return [float(var) for var in vars]


def main():
    pass


if __name__ == "__main__":
    main()


# End
