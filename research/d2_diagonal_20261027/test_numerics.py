import numpy as np
from numerics import base_grid, diagonal_bias, select_integer, time_factor


def test_deleting_cross_terms_can_make_bias_negative():
    # A PSD full quadratic plus linear term is an actual squared residual.
    a = np.array([[1., 1.]])
    h = a.T @ a
    f = np.array([[.75, .75]])
    linear = a[0]
    full = np.einsum('ti,ij,tj->t', f, h, f)-2*f@linear+1
    assert np.allclose(full, [.25])
    assert np.allclose(diagonal_bias(f, h, linear, 1), [-.875])


def test_diagonal_case_matches_full_and_ignores_zero_direction():
    f = np.array([[0., .2, .4], [0., .5, .6]])
    h = np.diag([99., 2., 3.])
    l = np.array([200., .4, -.1])
    assert np.allclose(diagonal_bias(f, h, l, 1),
                       np.einsum('ti,ij,tj->t', f, h, f)-2*f@l+1)


def test_search_keeps_integer_and_boundaries():
    grid = np.array([0, 2, 5, 10])
    obj = lambda t: (t-4.4)**2
    assert select_integer(grid, obj(grid), obj) == 4
    assert select_integer(grid, grid.astype(float), lambda t: t) == 0
    assert select_integer(grid, -grid.astype(float), lambda t: -t) == 10
    assert np.issubdtype(base_grid().dtype, np.integer)
    assert time_factor(0, 10) is None
    assert time_factor(20, 10) == 2
