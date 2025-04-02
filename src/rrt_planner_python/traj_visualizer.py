import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


class TrajectoryVisualizer:
    def __init__(self, dataset_file = None, dataset = None):
        # Load the dataset
        if dataset_file is not None:
            with open(dataset_file, "r") as f:
                self.dataset = json.load(f)
        else:
            self.dataset = dataset
    def plot_trajectory(self, trajectory, obstacles, start, goal, ax, plot_legend=True):
        """Plot a single trajectory with obstacles, start and goal points"""
        # Plot obstacles
        for obs in obstacles:
            circle = Circle(obs['center'], obs['radius'], color='gray', alpha=0.5)
            ax.add_patch(circle)

        # Plot trajectory
        trajectory = np.array(trajectory)
        if plot_legend:
            start_label = "Start"
            goal_label = "Goal"
        else:
            start_label = None
            goal_label = None
        ax.scatter([q[0] for q in trajectory], [q[1] for q in trajectory], s=15)
        ax.plot([q[0] for q in trajectory], [q[1] for q in trajectory], label="Trajectory", linewidth=2)

        # Plot start and goal points
        ax.scatter(start[0], start[1], color="green", s=100, label=start_label)
        ax.scatter(goal[0], goal[1], color="red", s=100, label=goal_label)    
        ax.set_title("Trajectory Visualization") 
        ax.set_xlabel("X")
        ax.set_ylabel("Y")   
        ax.grid(True)

    def visualize(self, trajectory_idx=None, bounds = None):
        """Visualize the trajectories"""
        fig, ax = plt.subplots(figsize=(6, 6))
        if bounds is not None:
            ax.set_xlim(bounds[0])
            ax.set_ylim(bounds[1])
        else:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        if trajectory_idx is None:
            # Visualize all trajectories
            for data in self.dataset:
                trajectory = data['trajectory']
                obstacles = data['obstacles']
                start = data['start']
                goal = data['goal']
                self.plot_trajectory(trajectory, obstacles, start, goal, ax)
        else:
            # Visualize a specific trajectory
            data = self.dataset[trajectory_idx]
            trajectory = data['trajectory']
            obstacles = data['obstacles']
            start = data['start']
            goal = data['goal']
            self.plot_trajectory(trajectory, obstacles, start, goal, ax)
        ax.legend(loc='upper right')
        plt.show()

    def find_trajectories_with_same_obstacles(self, obstacles):
        """
        Find all trajectories with the same obstacles (same number and dimensions)
        """
        # Sort obstacles by their center and radius to create a comparable structure
        sorted_obstacles = sorted(obstacles, key=lambda x: (tuple(x['center']), x['radius']))

        matching_trajectories = []

        for data in self.dataset:
            trajectory_obstacles = sorted(data['obstacles'], key=lambda x: (tuple(x['center']), x['radius']))
            if sorted_obstacles == trajectory_obstacles:
                matching_trajectories.append(data)
        return matching_trajectories

    def visualize_matching_trajectories(self, obstacles):
        """
        Visualize all trajectories that have the same obstacles
        """
        matching_trajectories = self.find_trajectories_with_same_obstacles(obstacles)
        
        if not matching_trajectories:
            print("No matching trajectories found.")
            return

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        print(f"Found {len(matching_trajectories)} matching trajectories.")
        # Plot all matching trajectories
        for i, data in enumerate(matching_trajectories):
            trajectory = data['trajectory']
            obstacles = data['obstacles']
            start = data['start']
            goal = data['goal']
            if i == 0:
                plot_legend = True
            else:
                plot_legend = False
            self.plot_trajectory(trajectory, obstacles, start, goal, ax, plot_legend=plot_legend)

        ax.legend(loc='upper right')
        plt.show()


if __name__ == "__main__":

    # Example usage
    visualizer = TrajectoryVisualizer("/home/arthur/Desktop/Code/conditional_diffusion_motion/examples/01_point_robot/trajectories/rrt_star_random_start_random_goal_200_samples.json")

    # Example obstacle configuration to match

    traj_example = visualizer.dataset[0]
    obstacles_to_match = traj_example['obstacles']
    # Visualize all trajectories with the same obstacles
    visualizer.visualize_matching_trajectories(obstacles_to_match)
