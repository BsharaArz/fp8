#!/usr/bin/env bash

source /shared_new/arzb/pyvenv/bin/activate
export NEURON_CC_FLAGS="--framework=XLA --model-type transformer --no-internal-hlo-remat --distribution-strategy=llm-training --dump=/shared_new/arzb/fp8/dump"
export JAX_TRACEBACK_FILTERING=off

/shared_new/arzb/pyvenv/bin/python /shared_new/arzb/fp8/experiments/train.py