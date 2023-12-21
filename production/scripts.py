"""Module for listing down additional custom functions required for production."""

import pandas as pd
import numpy as np
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin


def binned_selling_price(df):
    """Bin the selling price column using quantiles."""
    return pd.qcut(df["unit_price"], q=10)


class CustomScaler(BaseEstimator, TransformerMixin):
    """CustomScaler applies scaling to the input data and supports inverse scaling.

    The transformer scales the data using mean and standard deviation during fit.
    The scaled data can be inverse transformed back to the original data space.

    Parameters:
    -----------
    None

    Attributes:
    -----------
    mean_ : numpy.ndarray
        The mean value of each feature during fitting.
    std_ : numpy.ndarray
        The standard deviation of each feature during fitting.

    Example:
    --------
    >>> import numpy as np
    >>> from custom_scaler import CustomScaler
    >>> X_train = np.array([[1, 2], [3, 4], [5, 6]])
    >>> scaler = CustomScaler()
    >>> scaler.fit(X_train)
    CustomScaler()
    >>> X_scaled = scaler.transform(X_train)
    >>> X_scaled
    array([[-1.22474487, -1.22474487],
           [ 0.        ,  0.        ],
           [ 1.22474487,  1.22474487]])
    >>> X_inverse = scaler.inverse_transform(X_scaled)
    >>> np.allclose(X_train, X_inverse)
    True
    """

    def fit(self, X, y=None):
        """Compute the mean and standard deviation for scaling.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).

        Returns:
        --------
        self : CustomScaler
            Returns the instance of the CustomScaler.
        """
        X = check_array(X, copy=True, estimator=self)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        """Scale the input data.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data of shape (n_samples, n_features).

        Returns:
        --------
        X_scaled : numpy.ndarray
            Scaled data of shape (n_samples, n_features).
        """
        check_is_fitted(self)
        X = check_array(X, copy=True, estimator=self)
        X_scaled = (X - self.mean_) / self.std_
        return X_scaled

    def inverse_transform(self, X):
        """Inverse scale the input data.

        Parameters:
        -----------
        X : numpy.ndarray
            Scaled data of shape (n_samples, n_features).

        Returns:
        --------
        X_inverse : numpy.ndarray
            Original data of shape (n_samples, n_features).
        """
        check_is_fitted(self)
        X = check_array(X, copy=True, estimator=self)
        X_inverse = X * self.std_ + self.mean_
        return X_inverse

    def get_params(self, deep=True):
        """Get parameters for the estimator.

        Parameters:
        -----------
        deep : bool, default=True
            If True, will return the parameters for this estimator
            and contained subobjects that are estimators.

        Returns:
        --------
        params : dict
            Dictionary of parameter names mapped to their values.
        """
        return {}

    def set_params(self, **params):
        """Set the parameters of the estimator.

        Parameters:
        -----------
        **params : dict
            Estimator parameters.

        Returns:
        --------
        self : BaseEstimator
            Returns the estimator instance.
        """
        return self