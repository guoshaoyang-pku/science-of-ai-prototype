#!/usr/bin/env python3
"""Exact finite checks for the quadratic example in science_boundary_cases.md.

No training, network calls, random simulation, or generated files.
The continuum boundary follows from the algebra in the document; these checks
independently average the binomial distribution for selected positive h and B.
"""
from fractions import Fraction as F
from math import comb
import json


def expected_loss_by_enumeration(batch, curvature, step):
    result = F(0)
    for positive in range(batch + 1):
        mean_noise = F(2 * positive - batch, batch)
        probability = F(comb(batch, positive), 2 ** batch)
        loss = ((1 - step) ** 2 + curvature * (step * mean_noise) ** 2) / 2
        result += probability * loss
    return result


def sign(value):
    return (value > 0) - (value < 0)


def main():
    checked = 0
    examples = []
    for batch in (1, 2, 4, 8, 16, 64):
        for curvature in (1, batch + 1, batch + 2, batch + 3, 100):
            proxy_step = F(batch, batch + 1)
            optimal_step = F(batch, batch + curvature)
            proxy_gain = F(1, 2) - expected_loss_by_enumeration(batch, curvature, proxy_step)
            optimal_gain = F(1, 2) - expected_loss_by_enumeration(batch, curvature, optimal_step)
            assert proxy_gain == F(batch * (batch + 2 - curvature), 2 * (batch + 1) ** 2)
            assert optimal_gain == F(batch, 2 * (batch + curvature))
            assert sign(proxy_gain) == sign(batch + 2 - curvature)
            for multiplier in (F(1, 2), F(3, 2)):
                perturbed_gain = F(1, 2) - expected_loss_by_enumeration(
                    batch, curvature, optimal_step * multiplier
                )
                assert perturbed_gain < optimal_gain
            checked += 1
            if (batch, curvature) in ((1, 1), (1, 3), (1, 100), (4, 7)):
                examples.append({
                    "B": batch, "h": curvature,
                    "B_simple": 1, "B_noise": curvature,
                    "proxy_step": str(proxy_step),
                    "proxy_expected_loss_decrease": str(proxy_gain),
                    "optimal_step": str(optimal_step),
                    "optimal_expected_loss_decrease": str(optimal_gain),
                })
    print(json.dumps({
        "checks": checked,
        "method": "exact binomial enumeration using rational arithmetic",
        "claim_scope": "one expected population-loss update from x=(1,0); not full-trajectory convergence",
        "examples": examples,
    }, indent=2))


if __name__ == "__main__":
    main()
