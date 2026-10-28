"""Synthetic-only checks against direct GD; never access saved study data."""
import numpy as np
import pytest

import run as ablation


@pytest.mark.parametrize('outputs', [1, 2])
@pytest.mark.parametrize('rank_deficient', [False, True])
def test_spectral_ablation_matches_direct_feature_calculation(outputs, rank_deficient):
    rng = np.random.default_rng(20261027 + outputs + 10 * rank_deficient)
    n, m, width = 9, 13, 4
    phi = rng.normal(size=(n, width)) * .2
    if rank_deficient:
        phi[:, -1] = 0.
    val_phi = rng.normal(size=(m, width)) * .2
    data = {
        'phi': phi,
        'val_phi': val_phi,
        'train_x': rng.normal(size=(n, 4)),
        'val_x': rng.normal(size=(m, 4)),
    }
    model = ablation.legacy.spectral_model(data)
    terms = ablation.legacy.target_terms(
        model, data, 'vector' if outputs == 2 else 'interaction'
    )
    assert model['rank'] == width - int(rank_deficient)
    times = np.array([0, 1, 3, 17, 50])
    f, _ = ablation.legacy.response(model, times)
    calculated_diagonal = ablation.diagonal_bias(
        f, terms['h'], terms['linear'], terms['val_y2']
    )
    validation_projection = model['vectors'].T @ val_phi.T @ terms['vy'] / m
    explicit_diagonal = []
    for ft in f:
        own_terms = sum(
            ft[j] ** 2 * model['hval'][j, j] * np.mean(terms['g'][j] ** 2)
            for j in range(width)
        )
        linear_terms = -2 * sum(
            ft[j] * np.mean(terms['g'][j] * validation_projection[j])
            for j in range(width)
        )
        explicit_diagonal.append(own_terms + linear_terms + np.mean(terms['vy'] ** 2))
    np.testing.assert_allclose(calculated_diagonal, explicit_diagonal, rtol=0, atol=1e-12)

    bias, noise, _, _ = ablation.legacy.expected_curves(model, terms, times)
    weights = np.zeros((width, outputs))
    response = np.zeros((width, n))
    eta = ablation.legacy.SPEC['eta']
    sigma2 = .75
    for step in range(51):
        if step in times:
            index = int(np.where(times == step)[0][0])
            direct_bias = np.mean((val_phi @ weights - terms['vy']) ** 2)
            direct_noise = np.sum((val_phi @ response) ** 2) / m
            np.testing.assert_allclose(direct_bias, bias[index], rtol=0, atol=1e-12)
            np.testing.assert_allclose(direct_noise, noise[index], rtol=0, atol=1e-12)
            record = ablation.independent_feature_check(
                data, model, terms, step, sigma2, float(bias[index] + sigma2 * noise[index])
            )
            assert record['absolute_error'] < 1e-12
        if step < 50:
            weights -= 2 * eta * phi.T @ (phi @ weights - terms['y']) / n
            response -= 2 * eta * phi.T @ (phi @ response - np.eye(n)) / n
