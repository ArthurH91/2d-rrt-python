import unittest
import numpy as np
import torch
import json
import os
from rrt_planner_python.datasets import InterpolatedTrajectoryDataset
from rrt_planner_python.rrt_star import Obstacle  # Replace `your_module` with the actual module name

class TestInterpolatedTrajectoryDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a temporary JSON file for testing
        cls.test_json_file = "test_dataset.json"
        cls.test_data = [
            {
                "trajectory": [[0.1, 0.2], [0.3, 0.4]],
                "obstacles": [{"center": [0.25, 0.25], "radius": 0.05}],
                "start": [0.1, 0.2],
                "goal": [0.3, 0.4]
            },
            {
                "trajectory": [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
                "obstacles": [{"center": [0.75, 0.75], "radius": 0.05}],
                "start": [0.5, 0.6],
                "goal": [0.9, 1.0]
            }
        ]
        with open(cls.test_json_file, "w") as f:
            json.dump(cls.test_data, f)

    @classmethod
    def tearDownClass(cls):
        # Clean up the temporary JSON file
        os.remove(cls.test_json_file)

    def test_dataset_loading(self):
        """Test that the dataset is loaded correctly from the JSON file."""
        dataset = InterpolatedTrajectoryDataset(self.test_json_file)
        self.assertEqual(len(dataset), 2)  # Check the number of samples

    def test_interpolation(self):
        """Test that trajectories are interpolated to the correct length."""
        dataset = InterpolatedTrajectoryDataset(self.test_json_file)
        max_length = max(len(data["trajectory"]) for data in self.test_data)
        
        for i in range(len(dataset)):
            trajectory = dataset.interpolated_dataset[i]["trajectory"]
            self.assertEqual(len(trajectory), max_length)  # Check trajectory length

    def test_collision_free(self):
        """Test that interpolated trajectories are collision-free."""
        dataset = InterpolatedTrajectoryDataset(self.test_json_file)
        
        for i in range(len(dataset)):
            trajectory = dataset.interpolated_dataset[i]["trajectory"]
            obstacles = dataset.interpolated_dataset[i]["obstacles"]
            
            # Check that no point in the trajectory collides with any obstacle
            for point in trajectory:
                for obstacle in obstacles:
                    self.assertFalse(obstacle.contains(point))  # Check collision-free

    def test_noise_addition(self):
        """Test that noise is added correctly to trajectories."""
        dataset = InterpolatedTrajectoryDataset(self.test_json_file, noise_std=0.1)
        sample = dataset[0]
        
        # Check that the noisy trajectory is not equal to the clean trajectory
        self.assertFalse(torch.allclose(sample["noisy_trajectory"], sample["clean_trajectory"]))

        # Check that the noise is within the expected range
        noise = sample["noisy_trajectory"] - sample["clean_trajectory"]
        self.assertTrue(torch.all(noise.abs() <= 3 * 0.1))  # 3-sigma rule for Gaussian noise

    def test_tensor_conversion(self):
        """Test that the data is correctly converted to PyTorch tensors."""
        dataset = InterpolatedTrajectoryDataset(self.test_json_file)
        sample = dataset[0]
        
        # Check that all outputs are PyTorch tensors
        self.assertIsInstance(sample["noisy_trajectory"], torch.Tensor)
        self.assertIsInstance(sample["clean_trajectory"], torch.Tensor)
        self.assertIsInstance(sample["obstacles"], torch.Tensor)
        self.assertIsInstance(sample["start"], torch.Tensor)
        self.assertIsInstance(sample["goal"], torch.Tensor)

    def test_obstacle_contains(self):
        """Test the Obstacle.contains method."""
        obstacle = Obstacle(center=[0.5, 0.5], radius=0.1)
        
        # Test a point inside the obstacle
        self.assertTrue(obstacle.contains(np.array([0.5, 0.5])))
        
        # Test a point outside the obstacle
        self.assertFalse(obstacle.contains(np.array([0.7, 0.7])))

    def test_invalid_trajectory(self):
        """Test that a trajectory starting and finishing at the same place is excluded."""
        # Create test data with a trajectory that starts and ends at the same point
        invalid_data = [
            {
                "trajectory": [[0.1, 0.2], [0.15, 0.2]],  # Valid trajectory
                "obstacles": [{"center": [0.25, 0.25], "radius": 0.05}],
                "start": [0.1, 0.2],
                "goal": [0.15, 0.2]
            },
            {
                "trajectory": [[0.1, 0.2], [0.25, 0.25], [0.1, 0.2]],  # Invalid trajectory, start and end are the same
                "obstacles": [{"center": [0.25, 0.25], "radius": 0.05}],
                "start": [0.1, 0.2],
                "goal": [0.1, 0.2]
            }
        ]

        # Save the invalid data to a temporary JSON file
        invalid_json_file = "invalid_dataset.json"
        with open(invalid_json_file, "w") as f:
            json.dump(invalid_data, f)

        # Load the dataset
        dataset = InterpolatedTrajectoryDataset(invalid_json_file)

        # Check that the dataset does not contain the invalid trajectory
        self.assertEqual(len(dataset), 1)  # Only one valid trajectory should remain

        # Clean up the temporary JSON file
        os.remove(invalid_json_file)


if __name__ == "__main__":
    unittest.main()