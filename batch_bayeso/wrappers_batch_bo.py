import numpy as np
import time

from bayeso.utils import utils_logger

from batch_bayeso import BBOLocalPenalization


class BatchBayesianOptimization:
    def __init__(self, str_method, objective, bounds, size_batch, debug=False):
        self.str_method = str_method
        self.objective = objective
        self.bounds = bounds
        self.size_batch = size_batch
        self.debug = debug

        if self.str_method == 'local_penalization':
            self.model_bo = BBOLocalPenalization(self.bounds, self.size_batch)
        else:
            raise ValueError

        self.logger = utils_logger.get_logger(f'batch_bo-{str_method}')

    def get_initials(self, str_initial_method='sobol', seed=None):
        return self.model_bo.get_initials(str_initial_method, self.size_batch, seed=seed)

    def run_single_iteration(self, X, Y):
        X_batch, dict_info = self.model_bo.optimize(X, Y)
        time_overall = dict_info['time_overall']

        return X_batch, time_overall

    def run(self, num_iter, seed=None):
        X = self.get_initials(seed=seed)
        Y = self.objective(X)
        if self.debug:
            self.logger.debug('X.shape %s Y.shape %s', str(X.shape), str(Y.shape))

        if len(Y.shape) == 1:
            Y = Y[..., np.newaxis]

        times = np.array([0.0])

        for ind_iter in range(0, num_iter):
            X_batch, time_overall = self.run_single_iteration(X, Y)
            Y_batch = self.objective(X_batch)
            if len(Y_batch.shape) == 1:
                Y_batch = Y_batch[..., np.newaxis]

            X = np.concatenate([X, X_batch], axis=0)
            Y = np.concatenate([Y, Y_batch], axis=0)
            times = np.concatenate([times, [time_overall]], axis=0)

            self.logger.info('Iteration %d: X.shape %s Y.shape %s times.shape %s', ind_iter + 1, str(X.shape), str(Y.shape), str(times.shape))

            if self.debug:
                self.logger.debug('X_batch:\n%s', utils_logger.get_str_array(X_batch))

        return X, Y, times
