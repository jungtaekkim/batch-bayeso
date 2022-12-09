import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import scipy.spatial.distance as scisd

from bayeso.bo import base_bo
from bayeso import covariance
from bayeso import constants
from bayeso.gp import gp
from bayeso.gp import gp_kernel
from bayeso.utils import utils_bo
from bayeso.utils import utils_logger


class BaseBBO(base_bo.BaseBO):
    def __init__(self, range_X: np.ndarray,
        size_batch: int,
        str_cov: str=constants.STR_COV,
        str_acq: str=constants.STR_BO_ACQ,
        normalize_Y: bool=constants.NORMALIZE_RESPONSE,
        use_ard: bool=constants.USE_ARD,
        prior_mu: constants.TYPING_UNION_CALLABLE_NONE=None,
        str_optimizer_method_gp: str=constants.STR_OPTIMIZER_METHOD_GP,
        str_optimizer_method_bo: str=constants.STR_OPTIMIZER_METHOD_AO,
        str_modelselection_method: str=constants.STR_MODELSELECTION_METHOD,
        str_exp: str=None,
        debug: bool=False
    ):
        assert isinstance(range_X, np.ndarray)
        assert isinstance(size_batch, int)
        assert isinstance(str_cov, str)
        assert isinstance(str_acq, str)
        assert isinstance(normalize_Y, bool)
        assert isinstance(use_ard, bool)
        assert isinstance(str_optimizer_method_bo, str)
        assert isinstance(str_optimizer_method_gp, str)
        assert isinstance(str_modelselection_method, str)
        assert isinstance(str_exp, (type(None), str))
        assert isinstance(debug, bool)
        assert callable(prior_mu) or prior_mu is None
        assert len(range_X.shape) == 2
        assert range_X.shape[1] == 2
        assert (range_X[:, 0] <= range_X[:, 1]).all()
        assert str_cov in constants.ALLOWED_COV
        assert str_acq in constants.ALLOWED_BO_ACQ
        assert str_optimizer_method_gp in constants.ALLOWED_OPTIMIZER_METHOD_GP
        assert str_optimizer_method_bo in constants.ALLOWED_OPTIMIZER_METHOD_BO
        assert str_modelselection_method in constants.ALLOWED_MODELSELECTION_METHOD

        assert str_optimizer_method_bo == 'L-BFGS-B'

        str_surrogate = 'gp'
        assert str_surrogate in constants.ALLOWED_SURROGATE

        if str_exp is None:
            str_exp = 'batch'
        else:
            str_exp = f'batch_{str_exp}'

        super().__init__(range_X, str_surrogate, str_acq,
            str_optimizer_method_bo, normalize_Y, str_exp, debug)

        self.size_batch = size_batch
        self.str_cov = str_cov
        self.use_ard = use_ard
        self.str_optimizer_method_gp = str_optimizer_method_gp
        self.str_modelselection_method = str_modelselection_method
        self.prior_mu = prior_mu

    def _optimize(self, fun_negative_acquisition: constants.TYPING_CALLABLE,
        str_sampling_method: str,
        num_samples: int
    ) -> constants.TYPING_TUPLE_TWO_ARRAYS:
        list_next_point = []

        list_bounds = self._get_bounds()
        initials = self.get_samples(str_sampling_method,
            fun_objective=fun_negative_acquisition,
            num_samples=num_samples)

        for arr_initial in initials:
            next_point = minimize(
                fun_negative_acquisition,
                x0=arr_initial,
                bounds=list_bounds,
                method=self.str_optimizer_method_bo,
                options={'disp': False}
            )
            next_point_x = next_point.x
            list_next_point.append(next_point_x)
            if self.debug:
                self.logger.debug('acquired sample: %s',
                    utils_logger.get_str_array(next_point_x))

        next_points = np.array(list_next_point)
        next_point = utils_bo.get_best_acquisition_by_evaluation(
            next_points, fun_negative_acquisition)[0]
        return next_point, next_points

    def compute_posteriors(self,
        X_train: np.ndarray, Y_train: np.ndarray,
        X_test: np.ndarray, cov_X_X: np.ndarray,
        inv_cov_X_X: np.ndarray, hyps: dict
    ) -> np.ndarray:
        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(cov_X_X, np.ndarray)
        assert isinstance(inv_cov_X_X, np.ndarray)
        assert isinstance(hyps, dict)
        assert len(X_train.shape) == 2
        assert len(Y_train.shape) == 2
        assert len(X_test.shape) == 2
        assert len(cov_X_X.shape) == 2
        assert len(inv_cov_X_X.shape) == 2
        assert Y_train.shape[1] == 1
        assert X_train.shape[0] == Y_train.shape[0]
        assert X_test.shape[1] == X_train.shape[1] == self.num_dim
        assert cov_X_X.shape[0] == cov_X_X.shape[1] == X_train.shape[0]
        assert inv_cov_X_X.shape[0] == inv_cov_X_X.shape[1] == X_train.shape[0]

        pred_mean, pred_std, _ = gp.predict_with_cov(
            X_train, Y_train, X_test,
            cov_X_X, inv_cov_X_X, hyps, str_cov=self.str_cov,
            prior_mu=self.prior_mu, debug=self.debug
        )

        pred_mean = np.squeeze(pred_mean, axis=1)
        pred_std = np.squeeze(pred_std, axis=1)

        return pred_mean, pred_std
