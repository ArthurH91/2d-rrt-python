import unittest
import json
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
from unittest.mock import patch
from rrt_planner_python.traj_visualizer import TrajectoryVisualizer


class TestTrajectoryVisualizer(unittest.TestCase):

    def setUp(self):
        """Set up a temporary JSON file for the dataset"""
        # Create a dummy dataset
        self.dummy_data = [{
            "trajectory": [[-0.8, -0.8], [0.8, 0.8]],
            "obstacles": [{"center": [-0.2, 0.2], "radius": 0.2}],
            "start": [-0.8, -0.8],
            "goal": [0.8, 0.8]
        }]
        
        # Create a temporary file to store the dataset
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w', newline='')
        with open(self.temp_file.name, "w") as f:
            json.dump(self.dummy_data, f, indent=4)
    
    def tearDown(self):
        """Clean up the temporary file after the test"""
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_load_dataset(self):
        # Initialize visualizer with the path to the temporary dataset
        visualizer = TrajectoryVisualizer(self.temp_file.name)

        # Check if dataset is loaded correctly
        self.assertEqual(len(visualizer.dataset), 1)
        self.assertEqual(visualizer.dataset[0]["start"], [-0.8, -0.8])
        self.assertEqual(visualizer.dataset[0]["goal"], [0.8, 0.8])

    def test_find_trajectories_with_same_obstacles(self):
        # Define mock dataset
        dataset = [
            {
                "trajectory": [[-0.8, -0.8], [0.8, 0.8]],
                "obstacles": [{"center": [-0.2, 0.2], "radius": 0.2}],
                "start": [-0.8, -0.8],
                "goal": [0.8, 0.8]
            },
            {
                "trajectory": [[-0.6, -0.6], [0.7, 0.7]],
                "obstacles": [{"center": [-0.2, 0.2], "radius": 0.2}],
                "start": [-0.6, -0.6],
                "goal": [0.7, 0.7]
            },
            {
                "trajectory": [[-0.8, -0.8], [0.8, 0.8]],
                "obstacles": [{"center": [0.4, -0.4], "radius": 0.15}],
                "start": [-0.8, -0.8],
                "goal": [0.8, 0.8]
            }
        ]

        visualizer = TrajectoryVisualizer(self.temp_file.name)
        visualizer.dataset = dataset  # Use the mock dataset

        obstacles_to_match = [{"center": [-0.2, 0.2], "radius": 0.2}]
        matching_trajectories = visualizer.find_trajectories_with_same_obstacles(obstacles_to_match)

        self.assertEqual(len(matching_trajectories), 2)  # Two trajectories match
        self.assertEqual(matching_trajectories[0]["trajectory"], [[-0.8, -0.8], [0.8, 0.8]])
        self.assertEqual(matching_trajectories[1]["trajectory"], [[-0.6, -0.6], [0.7, 0.7]])

    @patch("matplotlib.pyplot.show")
    def test_visualize_matching_trajectories(self, mock_show):
        # Initialize the visualizer with the real dummy dataset
        visualizer = TrajectoryVisualizer(self.temp_file.name)

        # Use the obstacles from the first trajectory
        obstacles_to_match = visualizer.dataset[0]['obstacles']
        
        # Call the method to visualize matching trajectories
        visualizer.visualize_matching_trajectories(obstacles_to_match)

        # Verify that `plt.show()` was called (which renders the plot)
        mock_show.assert_called_once()

    def test_plot_trajectory(self):
        # Create a mock figure and axis
        fig, ax = plt.subplots()

        # Create a visualizer instance with a simple dataset
        dataset = [{
            "trajectory": [[-0.8, -0.8], [0.8, 0.8]],
            "obstacles": [{"center": [-0.2, 0.2], "radius": 0.2}],
            "start": [-0.8, -0.8],
            "goal": [0.8, 0.8]
        }]
        
        visualizer = TrajectoryVisualizer(self.temp_file.name)
        visualizer.dataset = dataset  # Use the mock dataset
        
        # Plot a single trajectory
        visualizer.plot_trajectory(dataset[0]["trajectory"], dataset[0]["obstacles"], dataset[0]["start"], dataset[0]["goal"], ax)

        # Check if elements are correctly plotted
        plotted_elements = ax.collections + ax.lines
        self.assertGreater(len(plotted_elements), 0)  # Ensure that something has been plotted

    @patch("matplotlib.pyplot.show")
    def test_empty_matching_trajectories(self, mock_show):
        # Test the case where no matching trajectories are found
        dataset = [{
            "trajectory": [[-0.8, -0.8], [0.8, 0.8]],
            "obstacles": [{"center": [-0.2, 0.2], "radius": 0.2}],
            "start": [-0.8, -0.8],
            "goal": [0.8, 0.8]
        }]
        
        visualizer = TrajectoryVisualizer(self.temp_file.name)
        visualizer.dataset = dataset  # Use the mock dataset
        
        obstacles_to_match = [{"center": [0.5, 0.5], "radius": 0.1}]  # Non-matching obstacles
        visualizer.visualize_matching_trajectories(obstacles_to_match)

        # Verify that plt.show() was not called, because no matching trajectories were found
        mock_show.assert_not_called()

if __name__ == "__main__":
    unittest.main()
