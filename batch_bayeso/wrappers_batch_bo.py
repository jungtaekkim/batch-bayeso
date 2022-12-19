import numpy as np
import time

from bayeso.utils import utils_logger

from batch_bayeso import BBORandom, BBOConstant, BBOPrediction, \
    BBOPureExploration, BBOLocalPenalization


class BatchBayesianOptimization:
    def __init__(self, str_method, objective, bounds, size_batch, debug=False):
        self.str_method = str_method
        self.objective = objective
        self.bounds = bounds
        self.size_batch = size_batch
        self.debug = debug

        # TODO: fix the following line
        debug_ = False

        if self.str_method == 'random':
            self.model_bo = BBORandom(self.bounds, self.size_batch, debug=debug_)
        elif self.str_method == 'constant':
            constant = 100.0
            self.model_bo = BBOConstant(self.bounds, self.size_batch, constant, debug=debug_)
        elif self.str_method == 'prediction':
            self.model_bo = BBOPrediction(self.bounds, self.size_batch, debug=debug_)
        elif self.str_method == 'pure_exploration':
            self.model_bo = BBOPureExploration(self.bounds, self.size_batch, debug=debug_)
        elif self.str_method == 'local_penalization':
            self.model_bo = BBOLocalPenalization(self.bounds, self.size_batch, debug=debug_)
        else:
            raise ValueError

        self.logger = utils_logger.get_logger(f'batch_bo-{str_method}')
        self.print_info()

    def print_info(self):
        num_separators = 15

        print('=' * num_separators + 'INFO START' + '=' * num_separators)
        print(f'bounds:\n{utils_logger.get_str_array(self.bounds)}')
        print(f'str_method: {self.str_method}')
        print(f'size_batch: {self.size_batch}')
        print(f'debug: {self.debug}')
        print('=' * num_separators + ' INFO END ' + '=' * num_separators)

    def get_initials(self, str_initial_method='sobol', seed=None):
        return self.model_bo.get_initials(str_initial_method, self.size_batch, seed=seed)

    def run_single_iteration(self, X, Y):
        X_batch, dict_info = self.model_bo.optimize(X, Y)
        time_overall = dict_info['time_overall']

        return X_batch, time_overall

    def run(self, num_iter, seed=None):
        time_first_start = time.time()

        X = self.get_initials(seed=seed)
        Y = self.objective(X)
        if self.debug:
            self.logger.debug('X.shape %s Y.shape %s', str(X.shape), str(Y.shape))

        if len(Y.shape) == 1:
            Y = Y[..., np.newaxis]

        time_first_end = time.time()
        times = np.array([time_first_end - time_first_start])

        for ind_iter in range(0, num_iter):
            X_batch, time_overall = self.run_single_iteration(X, Y)
            Y_batch = self.objective(X_batch)
            if len(Y_batch.shape) == 1:
                Y_batch = Y_batch[..., np.newaxis]

            X = np.concatenate([X, X_batch], axis=0)
            Y = np.concatenate([Y, Y_batch], axis=0)
            times = np.concatenate([times, [time_overall]], axis=0)
            assert X.shape[0] == Y.shape[0]
            assert times.shape[0] == (ind_iter + 2)

            self.logger.info('Iteration %d: X.shape %s Y.shape %s times.shape %s', ind_iter + 1, str(X.shape), str(Y.shape), str(times.shape))

            if self.debug:
                self.logger.debug('X_batch:\n%s', utils_logger.get_str_array(X_batch))

        return X, Y, times
