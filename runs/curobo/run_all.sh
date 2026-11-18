#!/bin/bash
# Replay every raw capture with both perceived-scene backends (voxel map on CPU, cuRobo TSDF/ESDF on GPU).
cd /home/builder/isaacsim_work/franka_factory
export OMP_NUM_THREADS=4
g=0
for s in intruder projectile bad_action nominal; do
  /home/builder/isaacsim/python.sh perception_replay.py runs/perception/${s}_raw.npz --backend voxel \
    --report runs/curobo/${s}_voxel.json > runs/curobo/${s}_voxel.log 2>&1 &
  PYTHONPATH=/home/builder/isaacsim_work/deps/site CUDA_VISIBLE_DEVICES=$g /home/builder/isaacsim/kit/python/bin/python3 \
    perception_replay.py runs/perception/${s}_raw.npz --backend curobo --no-baseline \
    --report runs/curobo/${s}_curobo.json > runs/curobo/${s}_curobo.log 2>&1 &
  g=$((1 - g))
done
wait
echo ALL_DONE
