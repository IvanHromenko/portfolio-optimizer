import numpy as np
import pytest

from quantum_portfolio.data import MarketData
from quantum_portfolio.ising import IsingModel, ising_energy, qubo_to_ising
from quantum_portfolio.qubo import build_qubo, evaluate_qubo, recommended_penalty


def test_ising_matches_qubo_for_all_four_asset_bitstrings():
    market_data = MarketData.sample_data()
    qubo = build_qubo(
        market_data,
        k=2,
        risk_aversion=0.5,
        penalty=recommended_penalty(market_data, 2, 0.5),
    )
    ising = qubo_to_ising(qubo)

    for bits in range(1 << ising.n_qubits):
        selection = np.array(
            [(bits >> index) & 1 for index in range(ising.n_qubits)]
        )
        spins = 1 - 2 * selection
        assert ising_energy(ising, spins) == pytest.approx(
            evaluate_qubo(qubo, selection)
        )


def test_diagonal_only_qubo_conversion_preserves_offset_and_energies():
    from quantum_portfolio.qubo import QUBOModel

    model = QUBOModel(
        matrix=np.diag([2.0, -4.0]),
        offset=3.5,
        penalty=1.0,
        required_assets=1,
    )
    ising = qubo_to_ising(model)

    assert ising.J == {}
    assert ising.offset == pytest.approx(2.5)
    assert ising.h == pytest.approx(np.array([-1.0, 2.0]))
    for spins in (np.array([-1, -1]), np.array([-1, 1]), np.array([1, -1]), np.array([1, 1])):
        selection = (1 - spins) // 2
        qubo_energy = evaluate_qubo(model, selection)
        assert ising_energy(ising, spins) == pytest.approx(qubo_energy)


def test_single_pair_interaction_has_hand_calculated_mapping():
    from quantum_portfolio.qubo import QUBOModel

    qubo = QUBOModel(
        matrix=np.array([[0.0, 2.0], [2.0, 0.0]]),
        offset=0.0,
        penalty=1.0,
        required_assets=0,
    )
    ising = qubo_to_ising(qubo)

    assert ising.h == pytest.approx([-1.0, -1.0])
    assert ising.J == {(0, 1): 1.0}
    assert ising.offset == pytest.approx(1.0)
    assert ising_energy(ising, np.array([-1, 1])) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "spins",
    [np.array([1]), np.array([0, 1]), np.array([1, -1, 1])],
)
def test_ising_energy_rejects_invalid_spins(spins):
    model = IsingModel(h=np.zeros(2), J={}, offset=0.0, n_qubits=2)
    with pytest.raises(ValueError):
        ising_energy(model, spins)


def test_ising_energy_rejects_invalid_coupling_indices():
    model = IsingModel(
        h=np.zeros(2),
        J={(1, 0): 1.0},
        offset=0.0,
        n_qubits=2,
    )
    with pytest.raises(ValueError):
        ising_energy(model, np.array([1, -1]))
