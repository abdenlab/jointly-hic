"""Exntension of sklearn.decomposition._BasePCA to support SVD decomposition."""

from numbers import Integral
from typing import ClassVar

import numpy as np
from scipy import sparse
from sklearn.base import _fit_context
from sklearn.decomposition._base import _BasePCA
from sklearn.utils import check_random_state, gen_batches
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils._param_validation import Interval, Real, StrOptions
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, svd_flip
from sklearn.utils.sparsefuncs import incr_mean_variance_axis


class SparseIncrementalSVD(_BasePCA):
    """Sparse Incremental Singular Value Decomposition (siSVD).

    Linear dimensionality reduction using Singular Value Decomposition of
    the data, keeping only the most significant singular vectors to
    project the data to a lower dimensional space.
    Derived from :class:`sklearn.decomposition.IncrementalPCA` and
    :class:`sklearn.decomposition.TruncatedSVD`.
    Unlike IncrementalPCA:
        * This estimator is designed to work on sparse `csr_matrix` input.
        * This estimator does NOT center the data before applying SVD.
    Unlike TruncatedSVD:
        * This estimator supports incremental (minibatch) learning via the
          `partial_fit` method.

    Parameters
    ----------
    n_components : int, default=None
        Number of components to keep. If ``n_components`` is ``None``,
        then ``n_components`` is set to **one less than**
        ``min(n_samples, n_features)``due to the way the ARPACK solver works.
    batch_size : int, default=None
        The number of samples to use for each batch. Only used when calling
        ``fit``. If ``batch_size`` is ``None``, then ``batch_size``
        is inferred from the data and set to ``5 * n_features``, to provide a
        balance between approximation accuracy and memory consumption.

    See Also
    --------
    sklearn.decomposition.IncrementalPCA
    sklearn.decomposition.TruncatedSVD

    """

    _parameter_constraints: ClassVar[dict] = {
        "n_components": [Interval(Integral, 1, None, closed="left"), None],
        "whiten": ["boolean"],
        "copy": ["boolean"],
        "batch_size": [Interval(Integral, 1, None, closed="left"), None],
        "algorithm": [StrOptions({"arpack", "randomized"})],
        "n_iter": [Interval(Integral, 0, None, closed="left")],
        "n_oversamples": [Interval(Integral, 1, None, closed="left")],
        "power_iteration_normalizer": [StrOptions({"auto", "OR", "LU", "none"})],
        "random_state": ["random_state"],
        "tol": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        n_components=None,
        *,
        batch_size=None,
        algorithm="randomized",
        n_iter=5,
        n_oversamples=10,
        power_iteration_normalizer="auto",
        random_state=None,
        tol=0.0,
        whiten=False,
        copy=True,
    ):
        """Initialize parameters for SparseIncrementalSVD."""
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.n_oversamples = n_oversamples
        self.power_iteration_normalizer = power_iteration_normalizer
        self.random_state = random_state
        self.tol = tol

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y=None):
        """Fit the model with X, using minibatches of size batch_size.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.components_ = None
        self.n_samples_seen_ = 0
        self.mean_ = 0.0
        self.var_ = 0.0
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

        # X = self._validate_params(
        #     X,
        #     accept_sparse=["csr", "csc", "lil"],
        #     copy=self.copy,
        #     dtype=[np.float64, np.float32],
        # )
        n_samples, n_features = X.shape

        if self.batch_size is None:
            self.batch_size_ = 5 * n_features
        else:
            self.batch_size_ = self.batch_size

        for batch in gen_batches(n_samples, self.batch_size_, min_batch_size=self.n_components or 0):
            X_batch = X[batch]
            self.partial_fit(X_batch)

        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y=None):
        """Incremental fit with X. All of X is processed as a single batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : Ignored
            Not used, present for API consistency by convention.
        check_input : bool, default=True
            Run check_array on X.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        random_state = check_random_state(self.random_state)
        if not hasattr(self, "components_"):
            self.components_ = None
        first_pass = self.components_ is None
        n_samples, n_features = X.shape

        if self.n_components is None:
            if self.components_ is None:
                self.n_components_ = min(n_samples, n_features) - 1
            else:
                self.n_components_ = self.components_.shape[0]
        elif not self.n_components <= n_features:
            raise ValueError(
                "n_components=%r invalid for n_features=%d, need more rows than columns for IncrementalPCA processing"
            )
        elif not self.n_components <= n_samples:
            raise ValueError("n_components=%r must be less or equal to the batch number of samples %d.")
        else:
            self.n_components_ = self.n_components

        if (self.components_ is not None) and (self.components_.shape[0] != self.n_components_):
            raise ValueError(
                "Number of input features has changed from %i "
                "to %i between calls to partial_fit! Try "
                "setting n_components to a fixed value."
            )

        # Convert X to sparse matrix, if not already
        X = sparse.csr_matrix(X)

        # This is the first partial_fit
        if first_pass:
            self.n_samples_seen_ = 0
            self.mean_ = np.zeros(X.shape[1])
            self.var_ = np.zeros(X.shape[1])

        col_mean, col_var, n_total_samples = incr_mean_variance_axis(
            X,
            axis=0,
            last_mean=self.mean_,
            last_var=self.var_,
            last_n=np.repeat(self.n_samples_seen_, X.shape[1]),
        )
        n_total_samples = n_total_samples[0]

        if self.n_samples_seen_ > 0:
            X = sparse.vstack(
                (
                    # The first `n_components` rows are the scaled singular vectors
                    # Each vector is length `n_features` (the previous basis)
                    self.singular_values_.reshape((-1, 1)) * self.components_,
                    # The new batch of data (n_samples, n_features) make up the remaining rows
                    X,
                    # Because we don't mean-center, we don't append a mean correction row
                )
            )

        if self.algorithm == "arpack":
            v0 = _init_arpack_v0(min(X.shape), random_state)
            U, S, Vt = sparse.linalg.svds(X, k=self.n_components_, tol=self.tol, v0=v0)
            S = S[::-1]
            U, Vt = svd_flip(U[:, ::-1], Vt[::-1])

        elif self.algorithm == "randomized":
            if self.n_components > X.shape[1]:
                raise ValueError(f"n_components({self.n_components}) must be <= n_features({X.shape[1]}).")
            U, S, Vt = randomized_svd(
                X,
                self.n_components,
                n_iter=self.n_iter,
                n_oversamples=self.n_oversamples,
                power_iteration_normalizer=self.power_iteration_normalizer,
                random_state=random_state,
            )

        explained_variance = S**2 / (n_total_samples - 1)
        explained_variance_ratio = S**2 / np.sum(col_var * n_total_samples)
        self.n_samples_seen_ = n_total_samples
        self.mean_ = col_mean
        self.var_ = col_var

        self.components_ = Vt
        self.singular_values_ = S
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance_ratio
        return self

    def transform(self, X):
        """Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set, using minibatches of size batch_size if X is
        sparse.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Projection of X in the first principal components.

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.decomposition import IncrementalPCA
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2],
        ...               [1, 1], [2, 1], [3, 2]])
        >>> ipca = IncrementalPCA(n_components=2, batch_size=3)
        >>> ipca.fit(X)
        IncrementalPCA(batch_size=3, n_components=2)
        >>> ipca.transform(X) # doctest: +SKIP

        """
        X_transformed = safe_sparse_dot(X, self.components_.T)
        return X_transformed
