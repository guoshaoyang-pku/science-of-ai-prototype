"""Exact finite-state counterexample to replacing coupled uncertainty by row minima.

Standard library only; no training, sampling, files, or imported optimizer.
The accompanying note proves the statements for every delta in [0, 1].
"""
from fractions import Fraction as F
from itertools import combinations
import json


def feasible(p, q, delta):
    return 0 <= p <= 1 and 0 <= q <= 1 and abs(p + q - 1) <= delta


def polygon_vertices(delta):
    # Intersect boundary lines a*p+b*q=c, then retain feasible intersections.
    lines = [(1, 0, 0), (1, 0, 1), (0, 1, 0), (0, 1, 1),
             (1, 1, 1 - delta), (1, 1, 1 + delta)]
    vertices = set()
    for (a, b, c), (d, e, f) in combinations(lines, 2):
        det = a * e - b * d
        if det:
            p, q = F(c * e - b * f, det), F(a * f - c * d, det)
            if feasible(p, q, delta):
                vertices.add((p, q))
    assert vertices
    return vertices


def main():
    rows = []
    for delta in (F(0), F(1, 10), F(1, 4), F(1, 2), F(3, 4), F(1)):
        vertices = polygon_vertices(delta)
        static_value = min((p + q) / 2 for p, q in vertices)
        rowwise_value = (min(p for p, _ in vertices) + min(q for _, q in vertices)) / 2
        assert static_value == (1 - delta) / 2
        assert rowwise_value == 0
        assert feasible(F(0), F(1), delta) and feasible(F(1), F(0), delta)
        assert feasible(F(0), F(0), delta) == (delta == 1)
        assert feasible(F(0), 1 - delta, delta)
        rows.append({"delta": str(delta), "static_value": str(static_value),
                     "rowwise_rectangular_value": str(rowwise_value),
                     "absolute_error": str(static_value - rowwise_value)})
    measures = []
    for epsilon in (F(0), F(1, 10), F(1, 4), F(1, 2), F(1)):
        # Error > epsilon iff delta < 1-2*epsilon; intersect with [0, 1].
        length = max(F(0), min(F(1), 1 - 2 * epsilon))
        if length:
            assert (1 - length / 2) / 2 > epsilon
            assert (1 - length) / 2 == epsilon
        measures.append({"epsilon": str(epsilon), "uniform_failure_measure": str(length)})
    print(json.dumps({"model": "static coupled transition uncertainty; terminal success reward",
                      "values": rows, "measures": measures}, indent=2))


if __name__ == "__main__":
    main()
