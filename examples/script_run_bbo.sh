#!/bin/bash

METHOD='local_penalization'
TARGET='branin'
SIZE_BATCH=5
NUM_ITER=10

python run_bbo.py --method $METHOD --target $TARGET --size_batch $SIZE_BATCH --num_iter $NUM_ITER
