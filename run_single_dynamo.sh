sim_cuda=0
ad_cuda=1

CUDA_VISIBLE_DEVICES=${sim_cuda} \
python closed_loop.py --scenario_path /home/yyin5/mack/datasets_mack_raw/HUGSIM/ss/scenarios/nuscenes/scene-0010-medium-01.yaml \
            --base_path ./configs/sim/nuscenes_base.yaml \
            --camera_path ./configs/sim/nuscenes_camera.yaml \
            --kinematic_path ./configs/sim/kinematic.yaml \
            --ad dynamo \
            --ad_cuda ${ad_cuda}
