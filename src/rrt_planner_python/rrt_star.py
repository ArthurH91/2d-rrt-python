import numpy as np
import matplotlib.pyplot as plt

class Obstacle:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

    def contains(self, point):
        """Check if a point is inside the obstacle"""
        return np.linalg.norm(point - self.center) <= self.radius

class RRTStar:
    def __init__(self, start, goal, bounds, obstacles=[], step_size=0.05, max_iter=1000, neighbor_radius=0.2):
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.bounds = bounds
        self.obstacles = obstacles
        self.step_size = step_size
        self.max_iter = max_iter
        self.neighbor_radius = neighbor_radius
        self.tree = {tuple(start): (None, 0)}  # Node -> (parent, cost)

    def sample(self):
        """Randomly sample a point within bounds"""
        while True:
            x = np.random.uniform(*self.bounds[0])
            y = np.random.uniform(*self.bounds[1])
            point = np.array([x, y])
            if not self.in_obstacle(point):
                return point

    def nearest(self, point):
        """Find the nearest node in the tree"""
        return min(self.tree.keys(), key=lambda node: np.linalg.norm(np.array(node) - point))

    def steer(self, from_node, to_point):
        """Move from `from_node` towards `to_point` by `step_size`"""
        direction = to_point - np.array(from_node)
        norm = np.linalg.norm(direction)
        if norm < self.step_size:
            new_point = to_point
        else:
            new_point = np.array(from_node) + self.step_size * (direction / norm)
        
        if self.in_obstacle(new_point) or self.path_collides(from_node, new_point):
            return None  # Invalid move
        return tuple(new_point)

    def in_obstacle(self, point):
        """Check if a point is inside any obstacle"""
        return any(obs.contains(point) for obs in self.obstacles)

    def path_collides(self, from_node, to_point, num_checks=10):
        """Check if the path between two nodes crosses an obstacle"""
        for i in range(1, num_checks + 1):
            interp_point = from_node + (to_point - np.array(from_node)) * (i / num_checks)
            if self.in_obstacle(interp_point):
                return True
        return False

    def reached_goal(self, node):
        """Check if we reached the goal"""
        return np.linalg.norm(np.array(node) - self.goal) < self.step_size

    def get_neighbors(self, new_node):
        """Find all nearby nodes within a given radius"""
        return [node for node in self.tree if np.linalg.norm(np.array(node) - np.array(new_node)) < self.neighbor_radius]

    def plan(self):
        """Run the RRT* algorithm"""
        for _ in range(self.max_iter):
            random_point = self.sample()
            nearest_node = self.nearest(random_point)
            new_node = self.steer(nearest_node, random_point)

            if new_node is None:
                continue  # Skip if invalid

            # Find neighbors within radius
            neighbors = self.get_neighbors(new_node)

            # Choose the best parent (min cost)
            best_parent = nearest_node
            best_cost = self.tree[nearest_node][1] + np.linalg.norm(np.array(new_node) - np.array(nearest_node))

            for neighbor in neighbors:
                cost = self.tree[neighbor][1] + np.linalg.norm(np.array(new_node) - np.array(neighbor))
                if cost < best_cost and not self.path_collides(neighbor, new_node):
                    best_parent = neighbor
                    best_cost = cost

            # Add new node to tree
            self.tree[new_node] = (best_parent, best_cost)

            # Rewire nearby nodes
            for neighbor in neighbors:
                new_cost = best_cost + np.linalg.norm(np.array(neighbor) - np.array(new_node))
                if new_cost < self.tree[neighbor][1] and not self.path_collides(new_node, neighbor):
                    self.tree[neighbor] = (new_node, new_cost)

            # Stop if we reach goal
            if self.reached_goal(new_node):
                return self.get_path(new_node)

        # Raise an exception if the goal is unreachable
        raise Exception("Goal is unreachable. RRT* failed to find a path.")

    def get_path(self, node):
        """Backtrack from goal to start"""
        path = [node]
        while node is not None:
            node = self.tree[node][0]
            if node is not None:
                path.append(node)
        return path[::-1]  # Reverse path

    def plot(self, path=None, show = True):
        """Plot the RRT* tree, obstacles, and path"""
        plt.figure(figsize=(6, 6))
        plt.xlim(self.bounds[0])
        plt.ylim(self.bounds[1])
        plt.scatter(*self.start, color="green", s=100, label="Start")
        plt.scatter(*self.goal, color="red", s=100, label="Goal")

        # Draw obstacles
        for obs in self.obstacles:
            circle = plt.Circle(obs.center, obs.radius, color="black", alpha=0.5)
            plt.gca().add_patch(circle)

        # Draw tree
        for node, (parent, _) in self.tree.items():
            if parent:
                plt.plot([node[0], parent[0]], [node[1], parent[1]], "b-", alpha=0.5)

        # Draw path if found
        if path:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], "r-", linewidth=2, label="Path")

        plt.legend()
        if show:
            plt.show()

    def get_obstacle_info(self):
        """Return obstacle information as a list of dictionaries"""
        return [{"center": obs.center.tolist(), "radius": obs.radius} for obs in self.obstacles]