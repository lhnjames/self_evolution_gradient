import numpy as np

from self_evolve.repair_space import (
    atom_weights_from_coordinates,
    circular_component_count,
    orthonormal_repair_basis,
    pareto_mask,
    sample_unit_directions,
    target_coordinates,
)


def test_repair_basis_is_orthonormal_in_atom_metric():
    gram = np.asarray([[1.0, 0.3], [0.3, 1.0]])
    basis = orthonormal_repair_basis(gram)
    observed = basis.atom_coefficients.T @ gram @ basis.atom_coefficients
    assert np.allclose(observed, np.eye(2), atol=1e-12)


def test_target_coordinates_and_atom_weights_preserve_dot_product():
    gram = np.asarray([[1.0, -0.2], [-0.2, 1.0]])
    basis = orthonormal_repair_basis(gram)
    source_target_dots = np.asarray([[0.4, -0.1], [0.2, 0.3]])
    coordinates = np.asarray([0.6, -0.8])
    atom_weights = atom_weights_from_coordinates(basis, coordinates)
    expected = atom_weights @ source_target_dots
    observed = coordinates @ target_coordinates(basis, source_target_dots)
    assert np.allclose(observed, expected, atol=1e-12)


def test_sampled_directions_have_unit_norm():
    directions = sample_unit_directions(4, 100, 7)
    assert directions.shape == (100, 4)
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)
    assert np.array_equal(sample_unit_directions(1, 50, 7), [[1.0], [-1.0]])


def test_circular_components_wrap_around():
    assert circular_component_count([True, True, False, False, True]) == 1
    assert circular_component_count([True, False, True, False]) == 2
    assert circular_component_count([False, False]) == 0
    assert circular_component_count([True, True]) == 1


def test_pareto_mask_keeps_gain_harm_frontier():
    mask = pareto_mask([1.0, 2.0, 1.5, 3.0], [0.0, 1.0, 2.0, 3.0])
    assert np.array_equal(mask, [True, True, False, True])
