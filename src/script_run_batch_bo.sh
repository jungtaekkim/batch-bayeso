#!/bin/bash

TARGET='branin'
SIZE_BATCH=5
NUM_ITER=20

python run_batch_bo.py --target $TARGET --size_batch $SIZE_BATCH --num_iter $NUM_ITER
