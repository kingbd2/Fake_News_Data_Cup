
import warnings
warnings.filterwarnings('ignore')
from multiprocessing import cpu_count

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, Lars, LassoLars, OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, SGDRegressor, PassiveAggressiveRegressor, RANSACRegressor, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import is_classifier


from local_modules.hunga_bunga.utilities import *
from local_modules.hunga_bunga.universal_params import *


linear_models_n_params = [
    (LinearRegression, normalize),

    (Ridge,
     {'alpha': alpha, 'normaliz': normalize, 'tol': tol,
      'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
      }),

    (Lasso,
     {'alpha': alpha, 'normalize': normalize, 'tol': tol, 'warm_start': warm_start
      }),

    (ElasticNet,
     {'alpha': alpha, 'normalize': normalize, 'tol': tol,
      'l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9],
      }),

    (Lars,
     {'normalize': normalize,
      'n_nonzero_coefs': [100, 300, 500, np.inf],
      }),

    (LassoLars,
     { 'max_iter_inf': max_iter_inf, 'normalize': normalize, 'alpha': alpha
      }),

    (OrthogonalMatchingPursuit,
     {'n_nonzero_coefs': [100, 300, 500, np.inf, None],
      'tol': tol, 'normalize': normalize
      }),

    (BayesianRidge,
     {
         'n_iter': [100, 300, 1000],
         'tol': tol, 'normalize': normalize,
         'alpha_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
         'alpha_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
         'lambda_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
         'lambda_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
     }),

    # WARNING: ARDRegression takes a long time to run
    (ARDRegression,
     {'n_iter': [100, 300, 1000],
      'tol': tol, 'normalize': normalize,
      'alpha_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
      'alpha_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
      'lambda_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
      'lambda_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
      'threshold_lambda': [1e2, 1e3, 1e4, 1e6]}),

    (SGDRegressor,
     {'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
      'penalty': penalty_12e, 'n_iter': n_iter, 'epsilon': epsilon, 'eta0': eta0,
      'alpha': [1e-6, 1e-5, 1e-2, 'optimal'],
      'l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9],
      'learning_rate': ['constant', 'optimal', 'invscaling'],
      'power_t': [0.1, 0.25, 0.5]
      }),

    (PassiveAggressiveRegressor,
     {'C': C, 'epsilon': epsilon, 'n_iter': n_iter, 'warm_start': warm_start,
      'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
      }),

    (RANSACRegressor,
     {'min_samples': [0.1, 0.5, 0.9, None],
      'max_trials': n_iter,
      'stop_score': [0.8, 0.9, 1],
      'stop_probability': [0.9, 0.95, 0.99, 1],
      'loss': ['absolute_loss', 'squared_loss']
      }),

    (HuberRegressor,
     { 'epsilon': [1.1, 1.35, 1.5, 2],
       'max_iter': max_iter, 'alpha': alpha, 'warm_start': warm_start, 'tol': tol
       }),

    (KernelRidge,
     {'alpha': alpha, 'degree': degree, 'gamma': gamma, 'coef0': coef0
      })
]

linear_models_n_params_small = [
    (LinearRegression, normalize),

    (Ridge,
     {'alpha': alpha_small, 'normalize': normalize
      }),

    (Lasso,
     {'alpha': alpha_small, 'normalize': normalize
      }),

    (ElasticNet,
     {'alpha': alpha, 'normalize': normalize,
      'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
      }),

    (Lars,
     {'normalize': normalize,
      'n_nonzero_coefs': [100, 300, 500, np.inf],
      }),

    (LassoLars,
     {'normalize': normalize, 'max_iter': max_iter_inf, 'alpha': alpha_small
      }),

    (OrthogonalMatchingPursuit,
     {'n_nonzero_coefs': [100, 300, 500, np.inf, None],
      'normalize': normalize
      }),

    (BayesianRidge,
     { 'n_iter': [100, 300, 1000],
       'alpha_1': [1e-6, 1e-3],
       'alpha_2': [1e-6, 1e-3],
       'lambda_1': [1e-6, 1e-3],
       'lambda_2': [1e-6, 1e-3],
       'normalize': normalize,
       }),

    # WARNING: ARDRegression takes a long time to run
    (ARDRegression,
     {'n_iter': [100, 300],
      'normalize': normalize,
      'alpha_1': [1e-6, 1e-3],
      'alpha_2': [1e-6, 1e-3],
      'lambda_1': [1e-6, 1e-3],
      'lambda_2': [1e-6, 1e-3],
      }),

    (SGDRegressor,
     {'loss': ['squared_loss', 'huber'],
      'penalty': penalty_12e, 'n_iter': n_iter,
      'alpha': [1e-6, 1e-5, 1e-2, 'optimal'],
      'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
      }),

    (PassiveAggressiveRegressor,
     {'C': C, 'n_iter': n_iter,
      }),

    (RANSACRegressor,
     {'min_samples': [0.1, 0.5, 0.9, None],
      'max_trials': n_iter,
      'stop_score': [0.8, 1],
      'loss': ['absolute_loss', 'squared_loss']
      }),

    (HuberRegressor,
     { 'max_iter': max_iter, 'alpha_small': alpha_small,
       }),

    (KernelRidge,
     {'alpha': alpha_small, 'degree': degree,
      })
]

svm_models_n_params_small = [
    (SVR,
     {'kernel': kernel, 'degree': degree, 'shrinking': shrinking
      }),

    (NuSVR,
     {'nu': nu_small, 'kernel': kernel, 'degree': degree, 'shrinking': shrinking,
      }),

    (LinearSVR,
     {'C': C_small, 'epsilon': epsilon,
      'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
      'intercept_scaling': [0.1, 1, 10]
      })
]

svm_models_n_params = [
    (SVR,
     {'C': C, 'epsilon': epsilon, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol, 'max_iter': max_iter_inf2
      }),

    (NuSVR,
     {'C': C, 'epsilon': epsilon, 'kernel': kernel, 'degree': degree, 'gamma': gamma, 'coef0': coef0, 'shrinking': shrinking, 'tol': tol, 'max_iter': max_iter_inf2
      }),

    (LinearSVR,
     {'C': C, 'epsilon': epsilon, 'tol': tol, 'max_iter': max_iter,
      'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
      'intercept_scaling': [0.1, 0.5, 1, 5, 10]
      })
]

neighbor_models_n_params = [
    (RadiusNeighborsRegressor,
     {'radius': neighbor_radius, 'algo': neighbor_algo, 'leaf_size': neighbor_leaf_size, 'metric': neighbor_metric,
      'weights': ['uniform', 'distance'],
      'p': [1, 2],
      }),

    (KNeighborsRegressor,
     {'n_neighbors': n_neighbors, 'algo': neighbor_algo, 'leaf_size': neighbor_leaf_size, 'metric': neighbor_metric,
      'p': [1, 2],
      'weights': ['uniform', 'distance'],
      })
]

gaussianprocess_models_n_params = [
    (GaussianProcessRegressor,
     {'kernel': [RBF(), ConstantKernel(), DotProduct(), WhiteKernel()],
      'n_restarts_optimizer': [3],
      'alpha': [1e-10, 1e-5],
      'normalize_y': [True, False]
      })
]

nn_models_n_params = [
    (MLPRegressor,
     { 'hidden_layer_sizes': [(16,), (64,), (100,), (32, 64)],
       'activation': ['identity', 'logistic', 'tanh', 'relu'],
       'alpha': alpha, 'learning_rate': learning_rate, 'tol': tol, 'warm_start': warm_start,
       'batch_size': ['auto', 50],
       'max_iter': [1000],
       'early_stopping': [True, False],
       'epsilon': [1e-8, 1e-5]
       })
]

nn_models_n_params_small = [
    (MLPRegressor,
     { 'hidden_layer_sizes': [(64,), (32, 64)],
       'activation': ['identity', 'tanh', 'relu'],
       'max_iter': [500],
       'early_stopping': [True],
       'learning_rate': learning_rate_small
       })
]

tree_models_n_params = [

    (DecisionTreeRegressor,
     {'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'min_impurity_split': min_impurity_split,
      'criterion': ['mse', 'mae']}),

    (ExtraTreesRegressor,
     {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
      'min_samples_leaf': min_samples_leaf, 'min_impurity_split': min_impurity_split, 'warm_start': warm_start,
      'criterion': ['mse', 'mae']}),

]

tree_models_n_params_small = [
    (DecisionTreeRegressor,
     {'max_features': max_features_small, 'max_depth': max_depth_small, 'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf,
      'criterion': ['mse', 'mae']}),

    (ExtraTreesRegressor,
     {'n_estimators': n_estimators_small, 'max_features': max_features_small, 'max_depth': max_depth_small, 'min_samples_split': min_samples_split,
      'min_samples_leaf': min_samples_leaf,
      'criterion': ['mse', 'mae']})
]


def gen_reg_data(x_mu=10., x_sigma=1., num_samples=100, num_features=3, y_formula=sum, y_sigma=1.):
    x = np.random.normal(x_mu, x_sigma, (num_samples, num_features))
    y = np.apply_along_axis(y_formula, 1, x) + np.random.normal(0, y_sigma, (num_samples,))
    return x, y

def run_all_regressors(x, y, small = True, normalize_x = True, n_jobs=cpu_count()-1, brain=False, test_size=0.2, n_splits=5, upsample=True, scoring=None, verbose=False):
    all_params = (linear_models_n_params_small if small else linear_models_n_params) + (nn_models_n_params_small if small else nn_models_n_params) + ([] if small else gaussianprocess_models_n_params) + neighbor_models_n_params + (svm_models_n_params_small if small else svm_models_n_params) + (tree_models_n_params_small if small else tree_models_n_params)
    return main_loop(all_params, StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=False, n_jobs=n_jobs, brain=brain)


class HungaBungaRegressor(RegressorMixin):
    def __init__(self, brain=False, test_size = 0.2, n_splits = 5, random_state=None, upsample=True, scoring=None, verbose=True, normalize_x = True, n_jobs =cpu_count() - 1):
        self.model = None
        self.brain = brain
        self.test_size = test_size
        self.n_splits = n_splits
        self.random_state = random_state
        self.upsample = upsample
        self.scoring = None
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.normalize_x = normalize_x
        super(HungaBungaRegressor, self).__init__()

    def fit(self, x, y):
        self.model = run_all_regressors(x, y, normalize_x=self.normalize_x, test_size=self.test_size, n_splits=self.n_splits, upsample=self.upsample, scoring=self.scoring, verbose=self.verbose, brain=self.brain, n_jobs=self.n_jobs)[0]

    def predict(self, x):
        return self.model.predict(x)


if __name__ == '__main__':
    x, y = gen_reg_data(10, 3, 100, 3, sum, 0.3)
    mdl = HungaBungaRegressor()
    mdl.fit(x, y)
    print(mdl.predict(x).shape)
