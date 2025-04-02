import json
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d

from rrt_planner_python.rrt_star import Obstacle


class InterpolatedTrajectoryDataset(Dataset):
    def __init__(self, json_file, noise_std=0.1):
        """
        Args:
            json_file (str): Path to the JSON file containing the dataset.
            noise_std (float): Standard deviation of Gaussian noise to add to trajectories.
        """
        # Load the dataset from the JSON file
        with open(json_file, "r") as f:
            self.dataset = json.load(f)

        # Interpolate all trajectories to the same length
        self.interpolate_trajectories()

        # Add noise to trajectories (optional)
        self.noise_std = noise_std

    def interpolate_trajectories(self):
        """Interpolate all trajectories to the same length."""
        # Find the maximum trajectory length
        self.max_length = max(len(data["trajectory"]) for data in self.dataset)

        # Interpolate all trajectories
        self.interpolated_dataset = []
        for data in self.dataset:
            trajectory = np.array(data["trajectory"], dtype=np.float32)
            obstacles = [Obstacle(obs["center"], obs["radius"]) for obs in data["obstacles"]]
            start = np.array(data["start"], dtype=np.float32)
            goal = np.array(data["goal"], dtype=np.float32)

            # Interpolate the trajectory
            trajectory_interp = self.interpolate_trajectory(trajectory, self.max_length)

            # Ensure the interpolated trajectory is collision-free
            if not self.is_collision_free(trajectory_interp, obstacles):
                print("Interpolated trajectory is not collision-free!")
                continue


            # Update the data with the interpolated trajectory
            self.interpolated_dataset.append({
                "trajectory": trajectory_interp,
                "obstacles": obstacles,
                "start": start,
                "goal": goal
            })

    def interpolate_trajectory(self, trajectory, target_length):
        """
        Interpolate a trajectory to a target length.
        
        Args:
            trajectory (np.ndarray): Input trajectory of shape (N, 2).
            target_length (int): Desired length of the trajectory.
        
        Returns:
            np.ndarray: Interpolated trajectory of shape (target_length, 2).
        """
        # Create interpolation functions for x and y coordinates
        x = np.linspace(0, 1, len(trajectory))
        f_x = interp1d(x, trajectory[:, 0], kind='linear')
        f_y = interp1d(x, trajectory[:, 1], kind='linear')
        
        # Generate new points
        x_new = np.linspace(0, 1, target_length)
        trajectory_interp = np.column_stack((f_x(x_new), f_y(x_new)))
        
        return trajectory_interp

    def is_collision_free(self, trajectory, obstacles):
        """
        Check if a trajectory is collision-free.
        
        Args:
            trajectory (np.ndarray): Trajectory of shape (N, 2).
            obstacles (list): List of Obstacle objects.
        
        Returns:
            bool: True if the trajectory is collision-free, False otherwise.
        """
        for point in trajectory:
            for obstacle in obstacles:
                if obstacle.contains(point):
                    return False
        return True

    def __len__(self):
        return len(self.interpolated_dataset)

    def __getitem__(self, idx):
        # Get the data point
        data_point = self.interpolated_dataset[idx]

        # Extract trajectory, obstacles, start, and goal
        trajectory = data_point["trajectory"]
        obstacles = data_point["obstacles"]
        start = data_point["start"]
        goal = data_point["goal"]

        # Add noise to the trajectory (optional)
        noisy_trajectory = self.add_noise(trajectory)

        # Convert to PyTorch tensors
        noisy_trajectory = torch.tensor(noisy_trajectory, dtype=torch.float32)
        clean_trajectory = torch.tensor(trajectory, dtype=torch.float32)
        start = torch.tensor(start, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)

        # Convert obstacles to a tensor
        obstacles_tensor = torch.tensor(
            [[obs.center[0], obs.center[1], obs.radius] for obs in obstacles],
            dtype=torch.float32
        )

        return {
            "noisy_trajectory": noisy_trajectory,
            "clean_trajectory": clean_trajectory,
            "obstacles": obstacles_tensor,
            "start": start,
            "goal": goal
        }

    def add_noise(self, trajectory):
        """Add Gaussian noise to a trajectory."""
        noise = np.random.normal(0, self.noise_std, trajectory.shape)
        return trajectory + noise