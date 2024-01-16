import time
import numpy as np
from scipy.stats import norm
import scipy.spatial.distance as scisd

from bayeso import constants
from bayeso.gp import gp_kernel
from bayeso.utils import utils_bo

from batch_bayeso import base_batch_bo


def grad_cov_matern52(X: np.ndarray, Xp:np.ndarray,
    lengthscales: constants.TYPING_UNION_ARRAY_FLOAT,
    signal: float
) -> np.ndarray:
    assert isinstance(X, np.ndarray)
    assert isinstance(Xp, np.ndarray)
    assert isinstance(lengthscales, (np.ndarray, float))
    assert len(X.shape) == 2
    assert len(Xp.shape) == 2
    if isinstance(lengthscales, np.ndarray):
        assert X.shape[1] == Xp.shape[1] == lengthscales.shape[0]
    else:
        assert X.shape[1] == Xp.shape[1]
    assert isinstance(signal, float)

    dist = scisd.cdist(X / lengthscales, Xp / lengthscales, metric='euclidean')
    dist = np.maximum(dist, 1e-8)
    grad_dist = np.transpose(((X / lengthscales**2)[:, None, :] - (Xp / lengthscales**2)[None, :, :]), (2, 0, 1)) * np.tile((1 / dist)[np.newaxis, ...], (X.shape[1], 1, 1))

    cov_X_Xp = signal**2 * (1.0 + np.sqrt(5.0) * dist + 5.0 / 3.0 * dist**2) \
        * np.exp(-1.0 * np.sqrt(5.0) * dist)

    grad_cov = cov_X_Xp * (-1.0 * np.sqrt(5.0)) + signal**2 * (np.sqrt(5.0) + 10.0 / 3.0 * dist) * np.exp(-1.0 * np.sqrt(5.0) * dist)
    grad_cov = grad_dist * grad_cov

    return grad_cov

def grad_cov_main(str_cov: str, X: np.ndarray, Xp: np.ndarray, hyps: dict,
    jitter: float=constants.JITTER_COV,
) -> np.ndarray:
    assert isinstance(str_cov, str)
    assert isinstance(X, np.ndarray)
    assert isinstance(Xp, np.ndarray)
    assert isinstance(hyps, dict)
    assert isinstance(jitter, float)
    assert str_cov in constants.ALLOWED_COV
    assert str_cov == 'matern52'

    grad_cov_X_Xp = grad_cov_matern52(X, Xp, hyps['lengthscales'], hyps['signal'])

    return grad_cov_X_Xp


class BBOLocalPenalization(base_batch_bo.BaseBBO):
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

        super().__init__(range_X, size_batch,
            str_cov=str_cov, str_acq=str_acq, normalize_Y=normalize_Y,
            use_ard=use_ard, prior_mu=prior_mu,
            str_optimizer_method_gp=str_optimizer_method_gp,
            str_optimizer_method_bo=str_optimizer_method_bo,
            str_modelselection_method=str_modelselection_method,
            str_exp=str_exp, debug=debug)

    def penalization(self, X: np.ndarray, X_batch: np.ndarray,
        r_batch: np.ndarray, s_batch: np.ndarray
    ) -> float:
        assert len(X.shape) == 2
        assert len(X_batch.shape) == 2
        assert X.shape[1] == X_batch.shape[1]
        assert len(r_batch.shape) == 1
        assert len(s_batch.shape) == 1
        assert r_batch.shape[0] == s_batch.shape[0] == X_batch.shape[0]

        vals = norm.cdf((np.sqrt(np.sum(np.square(X[:, None, :] - X_batch[None, :, :]), axis=-1)) - r_batch) / s_batch)
        assert np.all(vals.shape == (X.shape[0], X_batch.shape[0]))

        val = np.mean(vals, axis=1)
        return val

    def compute_acquisitions(self, X: np.ndarray,
        X_train: np.ndarray, Y_train: np.ndarray,
        X_batch: np.ndarray, minimum_evaluation: float, lipschitz: float,
        cov_X_X: np.ndarray, inv_cov_X_X: np.ndarray, hyps: dict
    ) -> np.ndarray:
        assert isinstance(X, np.ndarray)
        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)
        assert isinstance(X_batch, np.ndarray)
        assert isinstance(minimum_evaluation, float)
        assert isinstance(lipschitz, float)
        assert isinstance(cov_X_X, np.ndarray)
        assert isinstance(inv_cov_X_X, np.ndarray)
        assert isinstance(hyps, dict)
        assert len(X.shape) == 1 or len(X.shape) == 2
        assert len(X_train.shape) == 2
        assert len(Y_train.shape) == 2
        assert len(cov_X_X.shape) == 2
        assert len(inv_cov_X_X.shape) == 2
        assert Y_train.shape[1] == 1
        assert X_train.shape[0] == Y_train.shape[0]

        if len(X.shape) == 1:
            X = np.atleast_2d(X)

        assert X.shape[1] == X_train.shape[1] == self.num_dim
        assert cov_X_X.shape[0] == cov_X_X.shape[1] == X_train.shape[0]
        assert inv_cov_X_X.shape[0] == inv_cov_X_X.shape[1] == X_train.shape[0]

        fun_acquisition = utils_bo.choose_fun_acquisition(self.str_acq, hyps.get('noise', None))

        pred_mean, pred_std = self.compute_posteriors(
            X_train, Y_train, X,
            cov_X_X, inv_cov_X_X, hyps
        )

        acquisitions = fun_acquisition(
            pred_mean=pred_mean, pred_std=pred_std, Y_train=Y_train
        )
        acquisitions *= constants.MULTIPLIER_ACQ

        if X_batch.shape[0] > 0:
            pred_mean_batch, pred_std_batch = self.compute_posteriors(
                X_train, Y_train, X_batch,
                cov_X_X, inv_cov_X_X, hyps
            )
            pred_std_batch = np.maximum(pred_std_batch, 1e-8)

            r_batch = (pred_mean_batch - minimum_evaluation) / lipschitz
            s_batch = pred_std_batch / lipschitz

            acquisitions *= self.penalization(X, X_batch, r_batch, s_batch)

        return acquisitions

    def optimize(self, X_train: np.ndarray, Y_train: np.ndarray,
        str_sampling_method: str=constants.STR_SAMPLING_METHOD_AO,
        num_samples: int=constants.NUM_SAMPLES_AO,
        seed: int=None,
    ) -> constants.TYPING_TUPLE_ARRAY_DICT:
        assert isinstance(X_train, np.ndarray)
        assert isinstance(Y_train, np.ndarray)
        assert isinstance(str_sampling_method, str)
        assert isinstance(num_samples, int)
        assert isinstance(seed, (int, type(None)))
        assert len(X_train.shape) == 2
        assert len(Y_train.shape) == 2
        assert Y_train.shape[1] == 1
        assert X_train.shape[0] == Y_train.shape[0]
        assert X_train.shape[1] == self.num_dim
        assert num_samples > 0
        assert str_sampling_method in constants.ALLOWED_SAMPLING_METHOD

        time_start = time.time()
        Y_train_orig = Y_train

        if self.normalize_Y:
            if self.debug:
                self.logger.debug('Responses are normalized.')

            Y_train = utils_bo.normalize_min_max(Y_train)

        time_surrogate = 0.0
        time_acq = 0.0
        next_points_all = None
        acquisitions_all = None

        X_batch = []

        for ind in range(0, self.size_batch):
            time_start_surrogate = time.time()

            cov_X_X, inv_cov_X_X, hyps = gp_kernel.get_optimized_kernel(
                X_train, Y_train,
                self.prior_mu, self.str_cov,
                str_optimizer_method=self.str_optimizer_method_gp,
                str_modelselection_method=self.str_modelselection_method,
                use_ard=self.use_ard,
                debug=self.debug
            )

            time_end_surrogate = time.time()
            time_surrogate += (time_end_surrogate - time_start_surrogate)

            time_start_acq = time.time()

            minimum_evaluation = np.min(Y_train)

            # Jungtaek: let me fix this
            Xp = self.get_samples('uniform', num_samples=10000, seed=seed*(ind + 1) if seed is not None else seed)

            grad_cov_Xp_X = grad_cov_main(self.str_cov, Xp, X_train, hyps)

            d_mu = np.matmul(np.matmul(grad_cov_Xp_X, inv_cov_X_X), Y_train)
            d_mu = np.squeeze(d_mu, axis=2)

            norm_d_mu = np.linalg.norm(d_mu, axis=0)
            lipschitz = np.max(norm_d_mu)

            fun_negative_acquisition = lambda X_test: -1.0 * self.compute_acquisitions(
                X_test, X_train, Y_train, np.array(X_batch), minimum_evaluation, lipschitz, cov_X_X, inv_cov_X_X, hyps
            )
            next_point, next_points = self._optimize(fun_negative_acquisition,
                str_sampling_method=str_sampling_method, num_samples=num_samples, seed=seed + ind if seed is not None else seed)

            next_point = utils_bo.check_points_in_bounds(next_point[np.newaxis, ...], np.array(self._get_bounds()))[0]
            next_points = utils_bo.check_points_in_bounds(next_points, np.array(self._get_bounds()))

            time_end_acq = time.time()
            time_acq += (time_end_acq - time_start_acq)

            acquisitions = fun_negative_acquisition(next_points)

            X_batch.append(next_point)

            if next_points_all is None:
                next_points_all = next_points
            else:
                next_points_all = np.concatenate((next_points_all, next_points), axis=0)

            if acquisitions_all is None:
                acquisitions_all = acquisitions
            else:
                acquisitions_all = np.concatenate((acquisitions_all, acquisitions), axis=0)

        time_end = time.time()

        dict_info = {
            'next_points': next_points_all,
            'acquisitions': acquisitions_all,
            'Y_original': Y_train_orig,
            'Y_normalized': Y_train,
            'time_surrogate': time_surrogate,
            'time_acq': time_acq,
            'time_overall': time_end - time_start,
        }

        if self.debug:
            self.logger.debug('overall time consumed to acquire: %.4f sec.', time_end - time_start)

        return np.array(X_batch), dict_info
