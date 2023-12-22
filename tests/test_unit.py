import numpy as np
import sys
import os
module_folder = os.path.abspath('./production')
sys.path.append(module_folder)
from scripts import CustomScaler


def test_custom_scaler():
    # Test fit method
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = CustomScaler()
    fitted_scaler = scaler.fit(X_train)

    assert np.allclose(fitted_scaler.mean_, np.array([3., 4.]))
    assert np.allclose(fitted_scaler.std_, np.array([1.63299316, 1.63299316]))

    # Test transform method
    X_scaled = scaler.transform(X_train)
    assert np.allclose(X_scaled, np.array([[-1.22474487, -1.22474487],
                                           [0., 0.],
                                           [1.22474487, 1.22474487]]))

    # Test inverse_transform method
    X_inverse = scaler.inverse_transform(X_scaled)
    assert np.allclose(X_train, X_inverse)


if __name__ == "__main__":
    test_custom_scaler()
