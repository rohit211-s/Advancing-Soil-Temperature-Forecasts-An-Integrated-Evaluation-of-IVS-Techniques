

import numpy as np

from sklearn.utils.random import sample_without_replacement
from sklearn.utils.multiclass import type_of_target


__all__ = [
    'bootstrap_without_replacement',
    'complementary_pairs_bootstrap',
    'stratified_bootstrap'
]


def bootstrap_without_replacement(y, n_subsamples, random_state=None):

    n_samples = y.shape[0]
    return sample_without_replacement(n_samples, n_subsamples,
                                      random_state=random_state)


def complementary_pairs_bootstrap(y, n_subsamples, random_state=None):

    n_samples = y.shape[0]
    subsample = bootstrap_without_replacement(y, n_subsamples, random_state)
    complementary_subsample = np.setdiff1d(np.arange(n_samples), subsample)

    return subsample, complementary_subsample


def stratified_bootstrap(y, n_subsamples, random_state=None):

    type_of_target_y = type_of_target(y)
    allowed_target_types = ('binary', 'multiclass')
    if type_of_target_y not in allowed_target_types:
        raise ValueError(
            'Supported target types are: {}. Got {!r} instead.'.format(
                allowed_target_types, type_of_target_y))

    unique_y, y_counts = np.unique(y, return_counts=True)
    y_counts_relative = y_counts / y_counts.sum()
    y_n_samples = np.int32(np.round(y_counts_relative * n_subsamples))

    # the above should return grouped subsamples which approximately sum up
    # to n_subsamples but may not work out exactly due to rounding errors.
    # If this is the case, adjust the count of the largest class
    if y_n_samples.sum() != n_subsamples:
        delta = n_subsamples - y_n_samples.sum()
        majority_class = np.argmax(y_counts)
        y_n_samples[majority_class] += delta

    all_selected = np.array([], dtype=np.int32)
    for i, u in enumerate(unique_y):
        indices = np.where(y == u)[0]
        selected_indices = indices[bootstrap_without_replacement(indices,
                                                                 y_n_samples[i],
                                                                 random_state)]
        all_selected = np.concatenate((all_selected, selected_indices))

    return all_selected



import numpy as np

from scipy import sparse
from scipy.sparse import issparse

from sklearn.linear_model import LogisticRegression, Lasso
#from sklearn.linear_model.base import _preprocess_data
from sklearn.utils import check_X_y, check_random_state

__all__ = ['RandomizedLogisticRegression', 'RandomizedLasso']


def _rescale_data(X, weights):
    if issparse(X):
        size = weights.shape[0]
        weight_dia = sparse.dia_matrix((1 - weights, 0), (size, size))
        X_rescaled = X * weight_dia
    else:
        X_rescaled = X * (1 - weights)

    return X_rescaled


class RandomizedLogisticRegression(LogisticRegression):

    def __init__(self, weakness=0.5, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='liblinear', max_iter=100,
                 multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):
        self.weakness = weakness
        super(RandomizedLogisticRegression, self).__init__(
            penalty='l1', dual=False, tol=tol, C=C, fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling, class_weight=class_weight,
            random_state=random_state, solver=solver, max_iter=max_iter,
            multi_class=multi_class, verbose=verbose, warm_start=warm_start,
            n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):

        if not isinstance(self.weakness, float) or not (0.0 < self.weakness <= 1.0):
            raise ValueError('weakness should be a float in (0, 1], got %s' % self.weakness)

        X, y = check_X_y(X, y, accept_sparse='csr', dtype=[np.float64, np.float32],
                         order="C")

        n_features = X.shape[1]
        weakness = 1. - self.weakness
        random_state = check_random_state(self.random_state)

        weights = weakness * random_state.randint(0, 1 + 1, size=(n_features,))
        X_rescaled = _rescale_data(X, weights)
        return super(RandomizedLogisticRegression, self).fit(X_rescaled, y, sample_weight)


class RandomizedLasso(Lasso):

    def __init__(self, weakness=0.5, alpha=1.0, fit_intercept=True,
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        self.weakness = weakness
        super(RandomizedLasso, self).__init__(
            alpha=alpha, fit_intercept=fit_intercept,
            precompute=precompute, copy_X=copy_X,
            max_iter=max_iter, tol=tol, warm_start=warm_start,
            positive=positive, random_state=random_state,
            selection=selection)

    def fit(self, X, y):

        if not isinstance(self.weakness, float) or not (0.0 < self.weakness <= 1.0):
            raise ValueError('weakness should be a float in (0, 1], got %s' % self.weakness)

        X, y = check_X_y(X, y, accept_sparse=True)

        n_features = X.shape[1]
        weakness = 1. - self.weakness
        random_state = check_random_state(self.random_state)

        weights = weakness * random_state.randint(0, 1 + 1, size=(n_features,))

        # TODO: I am afraid this will do double normalization if set to true
        #X, y, _, _ = _preprocess_data(X, y, self.fit_intercept, normalize=self.normalize, copy=False,
        #             sample_weight=None, return_mean=False)

        # TODO: Check if this is a problem if it happens before standardization
        X_rescaled = _rescale_data(X, weights)
        return super(RandomizedLasso, self).fit(X_rescaled, y)


from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
#from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils.parallel import Parallel
from sklearn.utils.parallel import delayed
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import check_array, check_random_state, check_X_y, safe_mask
from sklearn.utils.validation import check_is_fitted


__all__ = ['StabilitySelection', 'plot_stability_path']

BOOTSTRAP_FUNC_MAPPING = {
    'subsample': bootstrap_without_replacement,
    'complementary_pairs': complementary_pairs_bootstrap,
    'stratified': stratified_bootstrap
}


def _return_estimator_from_pipeline(pipeline):
    """Returns the final estimator in a Pipeline, or the estimator
    if it is not"""
    if isinstance(pipeline, Pipeline):
        return pipeline._final_estimator
    else:
        return pipeline


def _bootstrap_generator(n_bootstrap_iterations, bootstrap_func, y,
                         n_subsamples, random_state=None):
    for _ in range(n_bootstrap_iterations):
        subsample = bootstrap_func(y, n_subsamples, random_state)
        if isinstance(subsample, tuple):
            for item in subsample:
                yield item
        else:
            yield subsample


def _fit_bootstrap_sample(base_estimator, X, y, lambda_name, lambda_value,
                          threshold=None):


    base_estimator.set_params(**{lambda_name: lambda_value})
    base_estimator.fit(X, y)

    # TODO: Reconsider if we really want to use SelectFromModel here or not
    selector_model = _return_estimator_from_pipeline(base_estimator)
    variable_selector = SelectFromModel(estimator=selector_model,
                                        threshold=threshold,
                                        prefit=True)
    return variable_selector.get_support()


def plot_stability_path(stability_selection, threshold_highlight=None,
                        **kwargs):

    check_is_fitted(stability_selection, 'stability_scores_')

    threshold = stability_selection.threshold if threshold_highlight is None else threshold_highlight
    paths_to_highlight = stability_selection.get_support(threshold=threshold)

    x_grid = stability_selection.lambda_grid / np.max(stability_selection.lambda_grid)
    print(type(x_grid))
    fig, ax = plt.subplots(1, 1, **kwargs)
    if not paths_to_highlight.all():
        ax.plot(x_grid[::-1], stability_selection.stability_scores_[~paths_to_highlight].T,
                'k:', linewidth=0.5)

    if paths_to_highlight.any():
        ax.plot(x_grid[::-1], stability_selection.stability_scores_[paths_to_highlight].T,
                'r-', linewidth=0.5)

    if threshold is not None:
        ax.plot(x_grid[::-1], threshold * np.ones_like(stability_selection.lambda_grid),
                'b--', linewidth=0.5)

    ax.set_ylabel('Stability score')
    ax.set_xlabel('Lambda / max(Lambda)')

    fig.tight_layout()

    return fig, ax


class StabilitySelection(BaseEstimator, TransformerMixin):

    def __init__(self, base_estimator=LogisticRegression(penalty='l2'), lambda_name='C',
                 lambda_grid=np.logspace(-5, -2, 25), n_bootstrap_iterations=100,
                 sample_fraction=0.5, threshold=0.6, bootstrap_func=bootstrap_without_replacement,
                 bootstrap_threshold=None, verbose=0, n_jobs=1, pre_dispatch='2*n_jobs',
                 random_state=None):
        self.base_estimator = base_estimator
        self.lambda_name = lambda_name
        self.lambda_grid = lambda_grid
        self.n_bootstrap_iterations = n_bootstrap_iterations
        self.sample_fraction = sample_fraction
        self.threshold = threshold
        self.bootstrap_func = bootstrap_func
        self.bootstrap_threshold = bootstrap_threshold
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.random_state = random_state

    def _validate_input(self):
        if not isinstance(self.n_bootstrap_iterations, int) or self.n_bootstrap_iterations <= 0:
            raise ValueError('n_bootstrap_iterations should be a positive integer, got %s' %
                             self.n_bootstrap_iterations)

        if not isinstance(self.sample_fraction, float) or not (0.0 < self.sample_fraction <= 1.0):
            raise ValueError('sample_fraction should be a float in (0, 1], got %s' % self.sample_fraction)

        if not isinstance(self.threshold, float) or not (0.0 < self.threshold <= 1.0):
            raise ValueError('threshold should be a float in (0, 1], got %s' % self.threshold)

        if self.lambda_name not in self.base_estimator.get_params().keys():
            raise ValueError('lambda_name is set to %s, but base_estimator %s '
                             'does not have a parameter '
                             'with that name' % (self.lambda_name,
                                                 self.base_estimator.__class__.__name__))

        if isinstance(self.bootstrap_func, str):
            if self.bootstrap_func not in BOOTSTRAP_FUNC_MAPPING.keys():
                raise ValueError('bootstrap_func is set to %s, but must be one of '
                                 '%s or a callable' %
                                 (self.bootstrap_func, BOOTSTRAP_FUNC_MAPPING.keys()))

            self.bootstrap_func = BOOTSTRAP_FUNC_MAPPING[self.bootstrap_func]
        elif not callable(self.bootstrap_func):
            raise ValueError('bootstrap_func must be one of %s or a callable' %
                             BOOTSTRAP_FUNC_MAPPING.keys())

    def fit(self, X, y):


        self._validate_input()

        X, y = check_X_y(X, y, accept_sparse='csr')

        n_samples, n_variables = X.shape
        n_subsamples = np.floor(self.sample_fraction * n_samples).astype(int)
        n_lambdas = self.lambda_grid.shape[0]

        base_estimator = clone(self.base_estimator)
        random_state = check_random_state(self.random_state)
        stability_scores = np.zeros((n_variables, n_lambdas))

        for idx, lambda_value in enumerate(self.lambda_grid):
            if self.verbose > 0:
                print("Fitting estimator for lambda = %.5f (%d / %d) on %d bootstrap samples" %
                      (lambda_value, idx + 1, n_lambdas, self.n_bootstrap_iterations))

            bootstrap_samples = _bootstrap_generator(self.n_bootstrap_iterations,
                                                     self.bootstrap_func, y,
                                                     n_subsamples, random_state=random_state)

            selected_variables = Parallel(
                n_jobs=self.n_jobs, verbose=self.verbose,
                pre_dispatch=self.pre_dispatch
            )(delayed(_fit_bootstrap_sample)(clone(base_estimator),
                                             X=X[safe_mask(X, subsample), :],
                                             y=y[subsample],
                                             lambda_name=self.lambda_name,
                                             lambda_value=lambda_value,
                                             threshold=self.bootstrap_threshold)
              for subsample in bootstrap_samples)

            stability_scores[:, idx] = np.vstack(selected_variables).mean(axis=0)

        self.stability_scores_ = stability_scores
        return self

    def get_support(self, indices=False, threshold=None):


        if threshold is not None and (not isinstance(threshold, float)
                                      or not (0.0 < threshold <= 1.0)):
            raise ValueError('threshold should be a float in (0, 1], '
                             'got %s' % self.threshold)

        cutoff = self.threshold if threshold is None else threshold
        mask = (self.stability_scores_.max(axis=1) > cutoff)

        return mask if not indices else np.where(mask)[0]

    def transform(self, X, threshold=None):

        X = check_array(X, accept_sparse='csr')
        mask = self.get_support(threshold=threshold)

        check_is_fitted(self, 'stability_scores_')

        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")

        if not mask.any():
            warn("No features were selected: either the data is"
                 " too noisy or the selection test too strict.",
                 UserWarning)
            return np.empty(0).reshape((X.shape[0], 0))

        return X[:, safe_mask(X, mask)]


base_estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomizedLasso(weakness=0.2))
    ])
selector = StabilitySelection(base_estimator=base_estimator, lambda_name='model__alpha',threshold=0.6,lambda_grid=np.logspace(-1, 1, num=100))
result=selector.fit(X, y)


fig, ax = plot_stability_path(result)
fig.show()

selected_variables = result.get_support(indices=True)
selected_scores = result.stability_scores_.max(axis=1)
print(selected_variables)
print('Selected variables are:')
print('-----------------------')

for idx, (variable, score) in enumerate(zip(selected_variables, selected_scores[selected_variables])):
    print('Variable %d: [%d], score %.3f' % (idx + 1, variable, score))
