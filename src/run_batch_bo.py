import numpy as np
import os
import time
import argparse

import bayeso_benchmarks.utils as bb_utils

from bbo import BBOLocalPenalization


def run_batch_bo_single_iteration(X, Y, bounds, size_batch):
    model_bo = BBOLocalPenalization(bounds, size_batch)

    X_batch, dict_info = model_bo.optimize(X, Y)
    time_overall = dict_info['time_overall']

    return X_batch, time_overall

def run_batch_bo(objective, bounds, size_batch):
    X = np.random.RandomState().uniform(size=(size_batch, bounds.shape[0])) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    Y = objective(X)
    times = np.array([0.0])

    print(X)

    for ind_iter in range(0, num_iter):
        X_batch, time_overall = run_batch_bo_single_iteration(X, Y, bounds, size_batch)
        print(f'Iteration {ind_iter + 1}')
        print('X_batch')
        print(X_batch)
        print('')

        X = np.concatenate([X, X_batch], axis=0)
        Y = np.concatenate([Y, objective(X_batch)], axis=0)
        times = np.concatenate([times, [time_overall]], axis=0)

    return X, Y, times


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--size_batch', type=int, required=True)
    parser.add_argument('--num_iter', type=int, required=True)

    args = parser.parse_args()

    str_target = args.target
    size_batch = args.size_batch
    num_iter = args.num_iter

    num_separators = 20

    list_str_target = str_target.split('_')
    if len(list_str_target) == 1:
        obj_target = bb_utils.get_benchmark(list_str_target[0])
    elif len(list_str_target) == 2:
        obj_target = bb_utils.get_benchmark(list_str_target[0], dim=int(list_str_target[1]))
    else:
        raise ValueError

    print('=' * num_separators + 'START' + '=' * num_separators)

    bounds = obj_target.get_bounds()
    print('bounds')
    print(bounds)

    def fun_target(X):
        Y = obj_target.output(X)
        return Y

    X, Y, times = run_batch_bo(fun_target, bounds, size_batch)

    print(f'X.shape {X.shape} Y.shape {Y.shape} times.shape {times.shape}')
    print('=' * num_separators + 'END' + '=' * num_separators)
