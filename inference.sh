set -x

id=$1

CUDA_VISIBLE_DEVICES=0 python phc/run_hydra.py project_name=OmniGrasp   exp_name=omnigrasp_Scene-${id}_track_15p_relative_rot learning=omnigrasp_rnn env=env_x_grab_z env.task=HumanoidOmniGraspZ env.motion_file=sample_data/Scene-${id}_sample_w_bg.pkl  env.models=['output/HumanoidIm/pulse_x_omnigrasp/Humanoid.pth']   env.numTrajSamples=20 env.trajSampleTimestepInv=15 robot=smplx_humanoid sim=hand_sim env.num_envs=768 learning.params.config.minibatch_size=4096 epoch=-1 env.use_track_reward=True env.use_track_reset=True env.use_track_obs=True +env.pregrasp_reward=False env.control_bodies=['Head','L_Toe','R_Toe','L_Wrist','R_Wrist','L_Index3','L_Middle3','L_Pinky3','L_Ring3','L_Thumb3','R_Index3','R_Middle3','R_Pinky3','R_Ring3','R_Thumb3'] +env.use_release_reward=True +env.background_urdf_name=Scene-${id}.urdf +env.target_object_name=Meter +env.humanoid_xml=Scene-${id}_smplx_humanoid.xml epoch=-1 test=True env.num_envs=1 headless=False #+env.draw_smpl_marker=True
