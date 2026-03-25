import numpy as np

from src.training.train_ml_strategy import _sigmoid


def test_sigmoid_is_finite_for_large_inputs() -> None:
    x = np.array([-1e12, -1e6, 0.0, 1e6, 1e12], dtype=float)
    y = _sigmoid(x)
    assert np.isfinite(y).all()
    assert (y >= 0.0).all()
    assert (y <= 1.0).all()

