#!/bin/bash
# Replay every capture with the cuRobo backend for one scene configuration.
# Usage: sweep.sh <tag> '<scene kwargs json>'
cd /home/builder/isaacsim_work/franka_factory
export OMP_NUM_THREADS=4 PYTHONPATH=/home/builder/isaacsim_work/deps/site
tag=$1
kwargs=${2:-"{}"}
mkdir -p runs/curobo/sweep/$tag
g=0
for s in intruder projectile bad_action nominal; do
  CUDA_VISIBLE_DEVICES=$g /home/builder/isaacsim/kit/python/bin/python3 perception_replay.py runs/perception/${s}_raw.npz \
    --backend curobo --no-baseline --scene-kwargs "$kwargs" \
    --report runs/curobo/sweep/$tag/${s}_curobo.json > runs/curobo/sweep/$tag/${s}_curobo.log 2>&1 &
  g=$((1 - g))
done
wait
