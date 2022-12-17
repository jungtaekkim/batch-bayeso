import argparse

import bayeso_benchmarks.utils as bb_utils

from batch_bayeso import BatchBayesianOptimization


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--size_batch', type=int, required=True)
    parser.add_argument('--num_iter', type=int, required=True)

    args = parser.parse_args()

    str_method = args.method
    str_target = args.target
    size_batch = args.size_batch
    num_iter = args.num_iter

    assert str_method in [
        'random',
        'constant',
        'prediction',
        'local_penalization',
    ]

    num_separators = 30

    list_str_target = str_target.split('_')
    if len(list_str_target) == 1:
        obj_target = bb_utils.get_benchmark(list_str_target[0])
    elif len(list_str_target) == 2:
        obj_target = bb_utils.get_benchmark(list_str_target[0], dim=int(list_str_target[1]))
    else:
        raise ValueError

    print('=' * num_separators + 'START' + '=' * num_separators)

    bounds = obj_target.get_bounds()

    def fun_target(X):
        Y = obj_target.output(X)
        return Y

    model_bbo = BatchBayesianOptimization(str_method, fun_target, bounds, size_batch, debug=True)
    X, Y, times = model_bbo.run(num_iter)

    print(f'X.shape {X.shape} Y.shape {Y.shape} times.shape {times.shape}')
    print('=' * num_separators + ' END ' + '=' * num_separators)
