import os.path as osp
import numpy as np
import pathlib
import json

from rrt_planner_python.traj_visualizer import TrajectoryVisualizer
from rrt_planner_python.datasets import InterpolatedTrajectoryDataset


file_name = "rrt_star_random_start_random_goal_2_samples"
dataset_path = osp.join((osp.dirname(__file__)), "trajectories", file_name + ".json")
data = json.load(open(dataset_path))

print(len(data))
# Example usage
visualizer = TrajectoryVisualizer(dataset_path)

# Example obstacle configuration to match
i = np.random.randint(0, len(data))
traj_example = visualizer.dataset[i]
obstacles_to_match = traj_example['obstacles']
# Visualize all trajectories with the same obstacles
visualizer.visualize_matching_trajectories(obstacles_to_match)
interpolated_dataset = InterpolatedTrajectoryDataset(dataset_path)


# # ### PLOT TRAJECTORIES ###
end_points = []  
dataset = []
obstacles = []
for trajs in interpolated_dataset:
    traj = trajs["clean_trajectory"]       
    end_points.append([traj[-1]])
    dataset.append(traj)
    obstacles.append(trajs["obstacles"][0][1].view(1,1))
    print(f"obs: {trajs["obstacles"][0][1].view(1,1)}")
### Dataset needs to have size [bs, seq_len, configuration_size]
dataset_ = np.asarray(dataset)
obstacles = np.asarray(obstacles)
print(dataset_.shape)
# print(obstacles.shape)
print(len(obstacles))
np.save(pathlib.Path(__file__).parent / "processed_datasets" / pathlib.Path(file_name + ".npy"), dataset)
np.save(pathlib.Path(__file__).parent / "processed_datasets" / pathlib.Path(file_name + "_obstacles.npy"), obstacles)
print(f"Dataset saved at {pathlib.Path(__file__).parent / 'processed_datasets'/ (file_name + '.npy')}")
# # # 