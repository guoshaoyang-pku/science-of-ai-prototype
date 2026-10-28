"""Synthetic-only tests; no study results or fitted stopping points are read."""
import numpy as np
import pytest

from sparse_numerics import budget_size, masked_h, pair_indices, rank_pairs, select_top


def test_pair_indices_use_only_active_modes_in_stable_order():
    expected = np.array([[1, 3], [1, 4], [3, 4]])
    np.testing.assert_array_equal(pair_indices([False, True, False, True, True]), expected)
    np.testing.assert_array_equal(pair_indices(np.array([4, 1, 3])), expected)
    assert pair_indices([]).shape == (0, 2)
    assert pair_indices([False, True]).shape == (0, 2)
    with pytest.raises(ValueError):
        pair_indices(np.array([1, 1]))


def test_change_score_matches_independent_finite_differences():
    rates = np.array([.07, .2, .6, 1.1])
    times = np.array([0., .2, 1., 3., 9.])
    h = np.array([[1., -.2, .3, .07], [-.2, 2., -.5, .1],
                  [.3, -.5, 3., -.09], [.07, .1, -.09, .5]])
    pairs = pair_indices(np.ones(4, dtype=bool))
    response = lambda t: -np.expm1(-np.outer(np.atleast_1d(t), rates)) / rates
    f = response(times)
    df = np.exp(-np.outer(times, rates))
    f_end = response([11.])[0]
    change, magnitude = rank_pairs(h, f, df, times, f_end, pairs)
    finite_difference_scores = []
    endpoint_scores = []
    for i, j in pairs:
        weighted_derivatives = []
        for t in times:
            epsilon = 1e-5 * max(1., t)
            plus, minus = response([t + epsilon, t - epsilon])
            plus_cross = 2 * h[i, j] * plus[i] * plus[j]
            minus_cross = 2 * h[i, j] * minus[i] * minus[j]
            weighted_derivatives.append(abs((plus_cross - minus_cross) / (2 * epsilon) * (t + 1)))
        finite_difference_scores.append(max(weighted_derivatives))
        endpoint_scores.append(abs(2 * h[i, j] * f_end[i] * f_end[j]))
    np.testing.assert_allclose(change, finite_difference_scores, rtol=2e-9, atol=1e-10)
    np.testing.assert_allclose(magnitude, endpoint_scores, rtol=0, atol=1e-14)


def test_masks_preserve_diagonal_symmetry_and_restore_full_matrix():
    h = np.array([[1., 2., -3.], [2., 4., 5.], [-3., 5., 6.]])
    original = h.copy()
    pairs = pair_indices(np.ones(3, dtype=bool))
    empty = masked_h(h, pairs, [])
    np.testing.assert_array_equal(empty, np.diag(np.diag(h)))
    partial = masked_h(h, pairs, np.array([1]))
    np.testing.assert_array_equal(partial, np.array([[1., 0., -3.], [0., 4., 0.], [-3., 0., 6.]]))
    np.testing.assert_array_equal(partial, partial.T)
    np.testing.assert_array_equal(masked_h(h, pairs, np.arange(len(pairs))), h)
    np.testing.assert_array_equal(h, original)


def test_strict_floor_budget_retains_no_pairs_for_rank_three_small_percentages():
    for percent in (0, 1, 5, 10, 20, 25):
        assert budget_size(3, percent) == 0
    assert budget_size(3, 100) == 3
    assert budget_size(129, 5) == 412
    assert budget_size(0, 100) == 0
    assert budget_size(1, 100) == 0
    with pytest.raises(TypeError):
        budget_size(3, 5.)


def test_top_selection_is_stable_and_depends_only_on_supplied_scores():
    scores = np.array([4., 7., 7., 1., 4.])
    np.testing.assert_array_equal(select_top(scores, 3), [1, 2, 0])
    np.testing.assert_array_equal(select_top(scores, 5), [1, 2, 0, 4, 3])
    assert select_top(scores, 0).size == 0
    assert select_top([], 0).size == 0
    with pytest.raises(ValueError):
        select_top(scores, 6)


def test_multioutput_averaging_is_already_contained_in_h():
    covariance = np.array([[1., .1, -.2], [.1, 2., .3], [-.2, .3, 3.]])
    g = np.array([[1., 2.], [-2., 1.], [.5, -1.]])
    h = covariance * (g @ g.T / g.shape[1])
    pairs = pair_indices(np.ones(3, dtype=bool))
    times = np.array([0., 1., 3.])
    rates = np.array([.1, .2, .3])
    f = -np.expm1(-np.outer(times, rates)) / rates
    df = np.exp(-np.outer(times, rates))
    change, magnitude = rank_pairs(h, f, df, times, f[-1], pairs)
    for k, (i, j) in enumerate(pairs):
        output_averaged_coefficient = sum(
            2 * covariance[i, j] * g[i, o] * g[j, o] for o in range(2)
        ) / 2
        expected = np.max(np.abs(output_averaged_coefficient
                         * (df[:, i] * f[:, j] + f[:, i] * df[:, j]) * (times + 1)))
        assert change[k] == pytest.approx(expected, abs=1e-14)
        assert magnitude[k] == pytest.approx(abs(output_averaged_coefficient * f[-1, i] * f[-1, j]), abs=1e-14)


def test_empty_pairs_and_chunked_scoring():
    empty = pair_indices([True])
    change, magnitude = rank_pairs(np.ones((1, 1)), np.ones((2, 1)),
                                   np.ones((2, 1)), [0, 1], [1], empty)
    assert change.shape == magnitude.shape == (0,)
    # >256 pairs exercises multiple chunks with a closed-form expected score.
    width = 25
    pairs = pair_indices(np.ones(width, dtype=bool))
    h = np.ones((width, width))
    change, magnitude = rank_pairs(h, np.ones((3, width)), np.ones((3, width)),
                                   [0, 1, 2], np.ones(width), pairs)
    np.testing.assert_array_equal(change, np.full(len(pairs), 12.))
    np.testing.assert_array_equal(magnitude, np.full(len(pairs), 2.))
