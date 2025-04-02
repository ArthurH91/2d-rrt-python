import json
import numpy as np
from rich.progress import (
    Progress,
    BarColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rrt_planner_python.rrt_star import (
    RRTStar,
    Obstacle,
)


# Define custom progress bar
progress_bar = Progress(
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)


# Function to generate a single trajectory
def generate_trajectory(start, goal, bounds, obstacles):
    try:
        rrt_star = RRTStar(start, goal, bounds, obstacles=obstacles)
        path = rrt_star.plan()
        trajectory = np.array(path)
        obstacle_info = rrt_star.get_obstacle_info()
        return {
            "trajectory": trajectory.tolist(),
            "obstacles": obstacle_info,
            "start": start.tolist(),
            "goal": goal.tolist(),
        }
    except Exception as e:
        print(f"Path planning failed with exception: {e}")
        return None


# Function to ensure start and goal are not in obstacles
def get_valid_point(bounds, obstacles):
    while True:
        point = np.random.uniform(bounds[0][0], bounds[0][1], size=2)        
        if not any(obs.contains(point) for obs in obstacles):
            return point


# Function to generate a dataset for multiple start/goal configurations
def generate_dataset(num_samples, n_traj_per_configuration, bounds, obstacles, random_start=False, random_goal=False):
    dataset = []
    start_goal_pairs = []  # To keep track of unique start/goal pairs

    with progress_bar as p:
        for _ in p.track(range(num_samples), description="Generating trajectories"):
            print(f"Generating trajectories for sample {_ + 1}/{num_samples}")
            
            # Get start and goal
            if random_start:
                start = get_valid_point(bounds, obstacles)
            else:
                start = np.array([-0.8, -0.8])
            if random_goal:
                goal = get_valid_point(bounds, obstacles)
            else:
                goal = np.array([0.8, 0.8])
            
            # Avoid duplicate start/goal pairs
            if (tuple(start), tuple(goal)) not in start_goal_pairs:
                start_goal_pairs.append((tuple(start), tuple(goal)))
                
                # Generate `n_traj_per_configuration` trajectories sequentially
                for _ in range(n_traj_per_configuration):
                    traj = generate_trajectory(start, goal, bounds, obstacles)
                    if traj:
                        dataset.append(traj)

    return dataset


if __name__ == "__main__":
    import argparse
    import os

    ### PARSER ###
    parser = argparse.ArgumentParser(description="Trajectory generation parser")

    parser.add_argument(
        "-n",
        "--num_trajs",
        type=int,
        default=10,
        help="Number of start/goal configurations",
    )
    parser.add_argument(
        "-rs",
        "--random_initial_start",
        action="store_true",
        help="Flag to use a random initial start",
    )
    parser.add_argument(
        "-rt",
        "--random_target",
        action="store_true",
        help="Flag to use a random target",
    )
    parser.add_argument(
        "-ro",
        "--random_obstacles",
        action="store_true",
        help="Flag to use random obstacles",
    )
    parser.add_argument(
        "-d",
        "--display-traj",
        action="store_true",
        help="Flag to display trajectories generated in matplotlib",
    )
    parser.add_argument(
        "-nt",
        "--n_traj_per_configuration",
        type=int,
        default=10,
        help="Number of trajectories to generate per start/goal pair",
    )
    args = parser.parse_args()

    # Define obstacles, start, goal, and bounds
    if not args.random_obstacles:
        obstacles = [
            Obstacle(center=(-0.2, 0.2), radius=0.2),
            Obstacle(center=(0.4, -0.5), radius=0.15),
            Obstacle(center=(-0.6, -0.3), radius=0.25),
            Obstacle(center=(0.5, 0.5), radius=0.15),
            Obstacle(center=(0.5, 0.0), radius=0.25),
        ]
    else:
        n_obs = np.random.randint(1, 6)
        obstacles = []
        for _ in range(n_obs):
            center = np.random.uniform(-1, 1, size=2)
            radius = np.random.uniform(0.1, 0.3)
            obstacles.append(Obstacle(center=center, radius=radius))

    bounds = [(-1, 1), (-1, 1)]

    # Generate dataset
    dataset = generate_dataset(
        args.num_trajs, args.n_traj_per_configuration, bounds, obstacles, args.random_initial_start, args.random_target
    )
    start_label = "random" if args.random_initial_start else "fixed"
    goal_label = "random" if args.random_target else "fixed"
    
    # Save dataset to a JSON file
    output_path = os.path.join("trajectories", f"rrt_star_{start_label}_start_{goal_label}_goal_{args.num_trajs}_samples.json")
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)
    
    if args.display_traj:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")

        for obs in obstacles:
            circle = Circle(obs.center, obs.radius, color="gray", alpha=0.5)
            ax.add_patch(circle)

        for data in dataset:
            trajectory = np.array(data["trajectory"])
            ax.plot(trajectory[:, 0], trajectory[:, 1], linewidth=2)

        ax.set_title("Trajectory Visualization")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)
        plt.show()
