"""Run directly with the project virtual environment."""
import numpy as np
from pack import project


def check_boundaries():
    history = np.zeros((7, 53))
    history[0,-5:] = [10, 20, 30, 10, 10]
    history[0,0] = 1  # ignored outside lookback
    history[1,-4:] = 10  # insufficient positive weeks
    history[2,-5:] = [10, 20, 30, 10, 1]  # unit quantum
    history[3,-5:] = [10, 20, 30, 10, 1.5]  # fractional history
    history[4,-5:] = [10, 20, 30, 10, 10]
    history[4,1] = 1  # first week inside the 52-week lookback
    history[5,-5:] = [12, 18, 6, 24, 30]  # quantum is six
    history[6,-5:] = [np.nan, 20, 30, 10, 10]
    forecast = np.tile([5., 15., 25., 3.25], (len(history),1))
    expected = forecast.copy()
    expected[0] = [0, 20, 20, 0]
    expected[5] = [6, 12, 24, 6]
    np.testing.assert_array_equal(project(history, forecast), expected)
    np.testing.assert_array_equal(project(history[:, :3], forecast), forecast)


if __name__ == "__main__":
    check_boundaries()
    print("Sales-increment boundary checks passed.")
