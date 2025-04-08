set -x

id=$1

tag=Scene-${id}
scp wfcluster:/projects_vol/gp_slab/fangzhou_new/Multi-Object/${tag}/processed/omnigrasp/${tag}_sample_w_bg.pkl sample_data
scp wfcluster:/projects_vol/gp_slab/fangzhou_new/Multi-Object/${tag}/processed/omnigrasp/${tag}.stl phc/data/assets/mesh/adt

# tag=scene_${id}
# mkdir -p output/HumanoidIm/omnigrasp_${tag}_track_15p_relative_rot
# scp wfcluster:/home/fangzhou.hong/Aria4D-scripts/submodules/Omnigrasp/output/HumanoidIm/omnigrasp_${tag}_track_15p_relative_rot/Humanoid.pth output/HumanoidIm/omnigrasp_${tag}_track_15p_relative_rot
