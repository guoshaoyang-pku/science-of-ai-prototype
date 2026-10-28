"""Exact two-state scope example; standard library only, no generated files."""
from fractions import Fraction as F
import json


Q = F(1, 10)
POLICIES = {"A": (F(2), F(9, 10)), "B": (F(1), F(1, 10))}


def solve_bellman(reward, p, gamma):
    # Solve (I - gamma P) v = (reward, 0) by the 2x2 determinant.
    a, b = 1 - gamma * (1 - p), -gamma * p
    c, d = -gamma * Q, 1 - gamma * (1 - Q)
    det = a * d - b * c
    v0, v1 = reward * d / det, -reward * c / det
    assert v0 == reward + gamma * ((1 - p) * v0 + p * v1)
    assert v1 == gamma * (Q * v0 + (1 - Q) * v1)
    return v0, v1


def main():
    averages = {}
    for name, (reward, p) in POLICIES.items():
        stationary = (Q / (p + Q), p / (p + Q))
        assert sum(stationary) == 1
        assert stationary[0] == stationary[0] * (1 - p) + stationary[1] * Q
        assert 0 < p < 1 and 0 < Q < 1
        averages[name] = stationary[0] * reward
    assert averages == {"A": F(1, 5), "B": F(1, 2)}

    rows = []
    for gamma in (F(0), F(1, 2), F(5, 8), F(9, 10), F(99, 100)):
        values = {}
        for name, (reward, p) in POLICIES.items():
            values[name] = solve_bellman(reward, p, gamma)[0]
            closed = reward * (1 - gamma + gamma * Q) / (
                (1 - gamma) * (1 - gamma + gamma * (p + Q))
            )
            assert values[name] == closed
        difference = values["A"] - values["B"]
        factored = (1 - F(9, 10) * gamma) * (1 - F(8, 5) * gamma) / (
            (1 - gamma) * (1 - F(4, 5) * gamma)
        )
        assert difference == factored
        assert (difference > 0) == (gamma < F(5, 8))
        assert (difference == 0) == (gamma == F(5, 8))
        rows.append({"gamma": str(gamma), **{k: str(v) for k, v in values.items()}})
    assert 1 - F(5, 8) == F(3, 8)
    print(json.dumps({"average_rewards": {k: str(v) for k, v in averages.items()},
                      "discounted_values_from_state_0": rows,
                      "ranking_boundary": "5/8",
                      "uniform_gamma_failure_measure_always_A": "3/8"}, indent=2))


if __name__ == "__main__":
    main()
