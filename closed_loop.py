import sys
import os
sys.path.append(os.getcwd())

import gymnasium
import hugsim_env
from argparse import ArgumentParser
from sim.utils.sim_utils import traj2control, traj_transform_to_global
import pickle
import json
import pickle
from sim.utils.launch_ad import launch, check_alive
from omegaconf import OmegaConf
import open3d as o3d
from sim.utils.score_calculator import hugsim_evaluate
import numpy as np
from moviepy import ImageSequenceClip

# Time spacing of the waypoints the AD side returns. The tracker needs it to place the plan on a
# timeline, and the scorer needs it to differentiate the plan into speeds/accelerations.
PLAN_TIMESTEP = 0.5

def to_video(observations, output_path):
    frames = []
    for obs in observations:
        row1 = np.concatenate([obs['CAM_FRONT_LEFT'], obs['CAM_FRONT'], obs['CAM_FRONT_RIGHT']], axis=1)
        row2 = np.concatenate([obs['CAM_BACK_RIGHT'], obs['CAM_BACK'], obs['CAM_BACK_LEFT']], axis=1)
        frame = np.concatenate([row1, row2], axis=0)
        frames.append(frame)
    clip = ImageSequenceClip(frames, fps=4)
    clip.write_videofile(output_path)


def create_gym_env(cfg, output):

    env = gymnasium.make('hugsim_env/HUGSim-v0', cfg=cfg, output=output)

    observations_save, infos_save = [], []
    obs, info = env.reset()
    done = False
    cnt = 0
    save_data = {'type': 'closeloop', 'frames': []}

    obs_pipe = os.path.join(output, 'obs_pipe')
    plan_pipe = os.path.join(output, 'plan_pipe')
    if not os.path.exists(obs_pipe):
        os.mkfifo(obs_pipe)
    if not os.path.exists(plan_pipe):
        os.mkfifo(plan_pipe)
    print('Ready for simulation')

    obs, info = None, None
    while not done:

        if obs is None or info is None:
            obs, info = env.reset()
        observations_save.append(obs['rgb'])
        infos_save.append(info)

        print(
            'ego x={:.3f}, y={:.3f}, speed={:.3f}, steer={:.3f}, lon_accel={:.3f}, lat_accel={:.3f}'.format(
                info["ego_box"][0],
                info["ego_box"][1],
                info["ego_velo"],
                info["ego_steer"],
                info["accelerate"],
                info["ego_velo"] * info["ego_velo"] * np.tan(-info['ego_steer']) / 2.7
            )
        )

        with open(obs_pipe, "wb") as pipe:
            pipe.write(pickle.dumps((obs, info)))
        with open(plan_pipe, "rb") as pipe:
            plan_traj = pickle.loads(pipe.read())

        if plan_traj is not None:
            # The plan is expressed in the ego frame at the *current* pose, so it has to be
            # anchored to that pose. Building the frame before env.step() keeps the recorded
            # ego_box, obj_boxes and time_stamp in the same frame of reference as the plan.
            imu_plan_traj = plan_traj[:, [1, 0]]
            imu_plan_traj[:, 1] *= -1
            global_traj = traj_transform_to_global(imu_plan_traj, info['ego_box'])
            frame = {
                'time_stamp': info['timestamp'],
                'is_key_frame': True,
                'ego_box': info['ego_box'],
                'obj_boxes': info['obj_boxes'],
                'obj_names': ['car' for _ in info['obj_boxes']],
                'planned_traj': {
                    'traj': global_traj,
                    'timestep': PLAN_TIMESTEP
                },
            }

            # The command is held for exactly one simulator step, so the tracker must be
            # discretized at cfg.kinematic.dt rather than at the AD's waypoint spacing.
            acc, steer_rate = traj2control(
                plan_traj, info, plan_dt=PLAN_TIMESTEP, sim_dt=cfg.kinematic.dt
            )
            # print(plan_traj, acc, steer_rate)

            action = {'acc': acc, 'steer_rate': steer_rate}
            obs, reward, terminated, truncated, info = env.step(action)
            cnt += 1
            done = terminated or truncated or cnt > 400

            # Episode-level bookkeeping: 'rc' is the progress reached by executing this plan,
            # so it stays post-step (the scorer only takes its maximum over the episode).
            frame['collision'] = info['collision']
            frame['rc'] = info['rc']
            save_data['frames'].append(frame)

        else:  # AD Side Crushed
            done = True

    with open(obs_pipe, "wb") as pipe:
        pipe.write(pickle.dumps('Done'))

    with open(os.path.join(output, 'data.pkl'), 'wb') as wf:
        pickle.dump([save_data], wf)
        
    to_video(observations_save, os.path.join(output, 'video.mp4'))
    with open(os.path.join(output, 'infos.pkl'), 'wb') as wf:
        pickle.dump(infos_save, wf)
    
    ground_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'ground.ply')).points)
    scene_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'scene.ply')).points)
    results = hugsim_evaluate([save_data], ground_xyz, scene_xyz)
    with open(os.path.join(output, 'eval.json'), 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--scenario_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--camera_path", type=str, required=True)
    parser.add_argument("--kinematic_path", type=str, required=True)
    parser.add_argument('--ad', default="uniad")
    parser.add_argument('--ad_cuda', default="1")
    args = parser.parse_args()

    scenario_config = OmegaConf.load(args.scenario_path)
    base_config = OmegaConf.load(args.base_path)
    camera_config = OmegaConf.load(args.camera_path)
    kinematic_config = OmegaConf.load(args.kinematic_path)
    cfg = OmegaConf.merge(
        {"scenario": scenario_config},
        {"base": base_config},
        {"camera": camera_config},
        {"kinematic": kinematic_config}
    )
    cfg.base.output_dir = cfg.base.output_dir + args.ad

    model_path = os.path.join(cfg.base.model_base, cfg.scenario.scene_name)
    model_config = OmegaConf.load(os.path.join(model_path, 'cfg.yaml'))
    model_config.model_path = model_path
    cfg.update(model_config)
    
    output = os.path.join(cfg.base.output_dir, cfg.scenario.scene_name+"_"+cfg.scenario.mode)
    os.makedirs(output, exist_ok=True)

    if args.ad == 'uniad':
        ad_path = cfg.base.uniad_path
    elif args.ad == 'vad':
        ad_path = cfg.base.vad_path
    elif args.ad == 'ltf':
        ad_path = cfg.base.ltf_path
    elif args.ad.startswith('dynamo'):
        ad_path = cfg.base.dynamo_path
    elif args.ad == 'gtrs':
        ad_path = cfg.base.gtrs_path
    elif args.ad == 'ztrs':
        ad_path = cfg.base.ztrs_path
    elif args.ad == 'gtrs_aug':
        ad_path = cfg.base.gtrs_aug_path
    else:
        raise NotImplementedError
    
    process = launch(ad_path, args.ad_cuda, output)
    try:
        create_gym_env(cfg, output)
        check_alive(process)
    except Exception as e:
        import traceback
        traceback.print_exc()
        process.kill()
    
    # For debug
    # create_gym_env(cfg, output)
