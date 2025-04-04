# Omnigrasp: Grasping Diverse Objects with Simulated Humanoids

Official implementation of NeurIPS 2024 paper: "Omnigrasp: Grasping Diverse Objects with Simulated Humanoids". In this project, we control a simulated humanoid to grasp diverse objects and follow diverse object trajectories. 

[[paper]](https://arxiv.org/abs/2407.11385) [[website]](https://zhengyiluo.github.io/Omnigrasp/) 

<div float="center">
  <img src="assets/omnigrasp_teaser.gif" />
</div>


## News 🚩


[December 12, 2024] Release inference code and models. 

## TODOs

- [ ] Release character drawing code. 
- [ ] Release training code (mainly the data processing part). 
- [x] Release inference code. 


### Dependencies

To create the environment, follow the following instructions: 

1. Create new conda environment and install pytorch:


```
conda create -n isaac python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirement.txt
```

2. Download and setup [Isaac Gym](https://developer.nvidia.com/isaac-gym). 


3. Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/download.php). Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. For SMPL, please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. For SMPLX, please download the v1.1 version. Rename The file structure should look like this:

```

|-- data
    |-- smpl
        |-- SMPLX_FEMALE.pkl
        |-- SMPLX_NEUTRAL.pkl
        |-- SMPLX_MALE.pkl

```


Make sure you have the SMPL-X paramters properly setup by running the following scripts:
```
python scripts/vis/vis_motion_mj.py
python scripts/joint_monkey_smpl.py
```

The SMPL model is used to adjust the height the humanoid robot to avoid penetnration with the ground during data loading. 

4. Use the following script to download trained models and sample data.

```
bash download_data.sh
```


## Evaluation 


### Viewer Shortcuts

| Keyboard | Function |
| ---- | --- |
| f | focus on humanoid |
| Right click + WASD | change view port |
| Shift + Right click + WASD | change view port fast |
| r | reset episode |
| j | apply large force to the humanoid |
| l | record screenshot, press again to stop recording|
| ; | cancel screen shot|
| m | cancel termination based on imitation |

... more shortcut can be found in `phc/env/tasks/base_task.py`


### Inference on GRAB objects + random trajectories

To evaluate a trained policy on the GRAB dataset, run the following script:

```
python phc/run_hydra.py  \
    project_name=OmniGrasp   exp_name=omnigrasp_neurips_grab  \
    learning=omnigrasp_rnn \
    env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15\
    robot=smplx_humanoid sim=hand_sim  \
    epoch=-1 test=True env.num_envs=1 headless=False
```

```
CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_from_scratch learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_from_pretrain_cut130 learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg_cut130.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 +env.table_remove_frame=43 +env.grasp_start_frame=43

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_from_pretrain_cut130_small_start25 learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg_cut130_small.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 +env.table_remove_frame=25 +env.grasp_start_frame=25

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_from_pretrain_cut130_mid learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg_cut130_mid.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 +env.table_remove_frame=25 +env.grasp_start_frame=25

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_from_pretrain_cut130_mid_track_head learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg_cut130_mid.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 +env.table_remove_frame=25 +env.grasp_start_frame=25 env.use_track_reward=True env.use_track_reset=True

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_from_pretrain_cut130_mid_new_track_head learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg_cut130_mid.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 +env.table_remove_frame=25 +env.grasp_start_frame=25 env.use_track_reward=True env.use_track_reset=True

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_from_pretrain_cut130_mid_new learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg_cut130_mid.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 +env.table_remove_frame=25 +env.grasp_start_frame=25

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_from_scratch_cut130_mid_new learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg_cut130_mid.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 +env.table_remove_frame=25 +env.grasp_start_frame=25

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_from_scratch_cut130_mid_new_track_head learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg_cut130_mid.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 +env.table_remove_frame=25 +env.grasp_start_frame=25 env.use_track_reward=True env.use_track_reset=True

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_tracking_v4 learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_obs=True env.use_track_reward=True env.use_grab_obs=False env.use_grab_reward=False env.use_track_reset=True env.use_grab_reset=False

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_from_scratch_cut130_mid_new_track_head_remove_table learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg_cut130_mid.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 +env.table_remove_frame=45 +env.grasp_start_frame=30 env.use_track_reward=True env.use_track_reset=True

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_tracking_v5_tp learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_obs=True env.use_track_reward=True env.use_grab_obs=False env.use_grab_reward=False env.use_track_reset=True env.use_grab_reset=False env.control_bodies=['Head','L_Wrist','R_Wrist']

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_from_pretrain_cut130_mid_new_track_tp_remove_table learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg_cut130_mid.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 +env.table_remove_frame=45 +env.grasp_start_frame=30 env.use_track_reward=True env.use_track_reset=True env.control_bodies=['Head','L_Wrist','R_Wrist']

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_from_scratch_cut130_mid_new_track_head_v2 learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg_cut130_mid.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 +env.table_remove_frame=45 +env.grasp_start_frame=25 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_from_scratch_cut130_mid_new_track_tp_no_pregrasp learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg_cut130_mid.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 +env.table_remove_frame=45 +env.grasp_start_frame=30 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['Head','L_Wrist','R_Wrist'] (env.grab_termination_distance=0.05)

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_from_scratch_cut130_mid_new_track_tp_no_pregrasp_release_reward learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg_cut130_mid.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 +env.table_remove_frame=45 +env.grasp_start_frame=25 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['Head','L_Wrist','R_Wrist'] +env.use_release_reward=True

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_from_scratch_cut130_mid_new_track_10p_no_pregrasp_release_reward learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg_cut130_mid.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 +env.table_remove_frame=45 +env.grasp_start_frame=25 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['Head','L_Wrist','R_Wrist','Pelvis','L_Hip','R_Hip','L_Shoulder','L_Elbow','R_Shoulder','R_Elbow'] +env.use_release_reward=True

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_otter_001_from_scratch_cut130_mid_new_track_tp_no_pregrasp_release_reward_v2 learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Otter_sample_w_bg_cut130_mid.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 +env.table_remove_frame=45 +env.grasp_start_frame=25 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['Head','L_Wrist','R_Wrist'] +env.use_release_reward=True

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_scene_003_from_scratch_cut130_mid_new_track_tp_no_pregrasp_release_reward learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Box1_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['Head','L_Toe','R_Toe','L_Wrist','R_Wrist'] +env.use_release_reward=True +env.background_urdf_name=Scene-003.urdf +env.target_object_name=Box1 env.episode_length=751
CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_scene_003_from_scratch_cut130_mid_new_track_tp_no_pregrasp_release_reward_copy learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Box1_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['Head','L_Toe','R_Toe','L_Wrist','R_Wrist'] +env.use_release_reward=True +env.background_urdf_name=Scene-003.urdf +env.target_object_name=Box1 env.episode_length=751

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_scene_003_from_scratch_cut130_mid_new_track_tp_no_pregrasp_release_reward_fix_contact_sensor learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Box1_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['Head','L_Wrist','R_Wrist','L_Toe','R_Toe'] +env.use_release_reward=True +env.background_urdf_name=Scene-003.urdf +env.target_object_name=Box1 env.episode_length=751


CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_scene_003_from_scratch_cut130_mid_new_track_tp_pretrain learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Box1_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_reward=False env.use_track_reset=False env.use_track_obs=False +env.pregrasp_reward=True env.control_bodies=['Head','L_Toe','R_Toe','L_Wrist','R_Wrist'] +env.use_release_reward=False +env.background_urdf_name=Scene-003.urdf +env.target_object_name=Box1 env.episode_length=751

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_scene_003_from_scratch_cut130_mid_new_track_tp_no_pregrasp_release_reward_track_finger learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Box1_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['Head','L_Toe','R_Toe','L_Wrist','R_Wrist','L_Index3','L_Middle3','L_Pinky3','L_Ring3','L_Thumb3','R_Index3','R_Middle3','R_Pinky3','R_Ring3','R_Thumb3'] +env.use_release_reward=True +env.background_urdf_name=Scene-003.urdf +env.target_object_name=Box1 env.episode_length=751
CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_scene_003_from_scratch_cut130_mid_new_track_tp_no_pregrasp_release_reward_track_finger_second_try learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Box1_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['Head','L_Toe','R_Toe','L_Wrist','R_Wrist','L_Index3','L_Middle3','L_Pinky3','L_Ring3','L_Thumb3','R_Index3','R_Middle3','R_Pinky3','R_Ring3','R_Thumb3'] +env.use_release_reward=True +env.background_urdf_name=Scene-003.urdf +env.target_object_name=Box1 env.episode_length=751
CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_scene_003_from_scratch_cut130_mid_new_track_tp_no_pregrasp_release_reward_track_finger_3rd_try learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Box1_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['Head','L_Toe','R_Toe','L_Wrist','R_Wrist','L_Index3','L_Middle3','L_Pinky3','L_Ring3','L_Thumb3','R_Index3','R_Middle3','R_Pinky3','R_Ring3','R_Thumb3'] +env.use_release_reward=True +env.background_urdf_name=Scene-003.urdf +env.target_object_name=Box1 env.episode_length=751

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_scene_003_from_scratch_cut130_mid_new_track_tp_no_pregrasp_release_reward_track_all_fingers learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Box1_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['Head','L_Toe','R_Toe','L_Wrist','R_Wrist','L_Index1','L_Index2','L_Index3','L_Middle1','L_Middle2','L_Middle3','L_Pinky1','L_Pinky2','L_Pinky3','L_Ring1','L_Ring2','L_Ring3','L_Thumb1','L_Thumb2','L_Thumb3','R_Index1','R_Index2','R_Index3','R_Middle1','R_Middle2','R_Middle3','R_Pinky1','R_Pinky2','R_Pinky3','R_Ring1','R_Ring2','R_Ring3','R_Thumb1','R_Thumb2','R_Thumb3'] +env.use_release_reward=True +env.background_urdf_name=Scene-003.urdf +env.target_object_name=Box1 env.episode_length=751

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_scene_003_from_scratch_cut130_mid_new_track_tp_no_pregrasp_release_reward_track_only_fingers learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Box1_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['L_Wrist','R_Wrist','L_Index3','L_Middle3','L_Pinky3','L_Ring3','L_Thumb3','R_Index3','R_Middle3','R_Pinky3','R_Ring3','R_Thumb3'] +env.use_release_reward=True +env.background_urdf_name=Scene-003.urdf +env.target_object_name=Box1 env.episode_length=751

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_scene_003_from_scratch_cut130_mid_new_track_tp_no_pregrasp_release_reward_track_only_right_hand learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Box1_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['R_Wrist','R_Index3','R_Middle3','R_Pinky3','R_Ring3','R_Thumb3'] +env.use_release_reward=True +env.background_urdf_name=Scene-003.urdf +env.target_object_name=Box1 env.episode_length=751

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_scene_003_from_scratch_cut130_mid_new_track_tp_no_pregrasp_release_reward_track_only_right_hand_krot100 learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Box1_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['R_Wrist','R_Index3','R_Middle3','R_Pinky3','R_Ring3','R_Thumb3'] +env.use_release_reward=True +env.background_urdf_name=Scene-003.urdf +env.target_object_name=Box1 env.episode_length=751 env.reward_specs.k_rot=100


CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_scene_003_track_finger_multiply_reward learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Box1_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['Head','L_Toe','R_Toe','L_Wrist','R_Wrist','L_Index3','L_Middle3','L_Pinky3','L_Ring3','L_Thumb3','R_Index3','R_Middle3','R_Pinky3','R_Ring3','R_Thumb3'] +env.use_release_reward=True +env.background_urdf_name=Scene-003.urdf +env.target_object_name=Box1 env.episode_length=751

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_scene_003_08x_track_15p learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Box1_08x_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['Head','L_Toe','R_Toe','L_Wrist','R_Wrist','L_Index3','L_Middle3','L_Pinky3','L_Ring3','L_Thumb3','R_Index3','R_Middle3','R_Pinky3','R_Ring3','R_Thumb3'] +env.use_release_reward=True +env.background_urdf_name=Scene-003_08x.urdf +env.target_object_name=Box1_08x env.episode_length=751

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_scene_005_track_15p learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Cookie_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['Head','L_Toe','R_Toe','L_Wrist','R_Wrist','L_Index3','L_Middle3','L_Pinky3','L_Ring3','L_Thumb3','R_Index3','R_Middle3','R_Pinky3','R_Ring3','R_Thumb3'] +env.use_release_reward=True +env.background_urdf_name=Scene-005.urdf +env.target_object_name=Cookie env.episode_length=364

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_scene_004_from_scratch_cut130_mid_new_track_tp_no_pregrasp_release_reward learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Bottle_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['Head','L_Toe','R_Toe','L_Wrist','R_Wrist'] +env.use_release_reward=True +env.background_urdf_name=Scene-004.urdf +env.target_object_name=Bottle env.episode_length=1454

epoch=-1 test=True env.num_envs=1 headless=False +env.draw_smpl_marker=True


```

Rendering the above trajectory might be slow, to speed up, use less number of environments or disable object trajectory rendering by press "k". 


### Tracing words in air 

[Coming soon]


## Training 

[This section is under construction]

### Data processing:

To train Omnigrasp, we need to process the GRAB dataset. Notice that while Omnigrasp does not depend on the GRAB dataset (we train on oakink and OMOMO as well), the GRAB dataset is used as a pre-grasp provider and inital pose provider. When training on oakink/OMOMO, a single sample motion from GRAB is used as the initial pose. 

To setup the GRAB dataset, please follow the following instructions:

1. Download the GRAB dataset from [here](https://grab.is.tue.mpg.de/).
2. Run the following script to process the dataset:

```
python scripts/data_process/convert_grab_smplx.py
```

