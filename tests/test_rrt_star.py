import unittest
import numpy as np
from rrt_planner_python.rrt_star import RRTStar, Obstacle

class TestRRTStar(unittest.TestCase):

    def test_sample_within_bounds(self):
        """Test that sampled points are within the specified bounds""" 
        rrt_star = RRTStar(start=(0, 0), goal=(1, 1), bounds=[(-1, 1), (-1, 1)])
        for _ in range(100):
            point = rrt_star.sample()
            self.assertTrue(-1 <= point[0] <= 1)
            self.assertTrue(-1 <= point[1] <= 1)

    def test_obstacle_collision(self):
        """Test that points inside obstacles are not sampled"""
        obstacles = [Obstacle(center=(0, 0), radius=0.5)]
        rrt_star = RRTStar(start=(0, 0), goal=(1, 1), bounds=[(-1, 1), (-1, 1)], obstacles=obstacles)
        for _ in range(100):
            point = rrt_star.sample()
            self.assertTrue(np.linalg.norm(point - np.array([0, 0])) > 0.5)

    def test_reached_goal(self):
        """Test that the goal is reached correctly with tolerance"""
        rrt_star = RRTStar(start=(0, 0), goal=(0.05, 0.05), bounds=[(-1, 1), (-1, 1)], step_size=0.1)
        path = rrt_star.plan()
        self.assertIsNotNone(path)
        final_point = path[-1]
        # Allow a small tolerance since the path may not end exactly at (0.05, 0.05)
        self.assertTrue(np.linalg.norm(np.array(final_point) - np.array([0.05, 0.05])) < rrt_star.step_size)

    def test_reached_goal_unreachable(self):
        """Test that an exception is raised if the goal is unreachable"""
        obstacles = [Obstacle(center=(0, 0), radius=1)]
        rrt_star = RRTStar(start=(-1, -1), goal=(1, 1), bounds=[(-1, 1), (-1, 1)], obstacles=obstacles)
        with self.assertRaises(Exception):
            rrt_star.plan()

    def test_path_collision_free(self):
        """Test that the generated path does not collide with obstacles"""
        obstacles = [Obstacle(center=(0, 0), radius=0.2)]
        rrt_star = RRTStar(start=(-0.8, -0.8), goal=(0.8, 0.8), bounds=[(-1, 1), (-1, 1)], obstacles=obstacles, max_iter=1000)
        
        try:
            path = rrt_star.plan()
            self.assertIsNotNone(path)
            
            # Ensure no collisions along the path
            for i in range(len(path) - 1):
                self.assertFalse(rrt_star.path_collides(path[i], path[i + 1]))
        except Exception as e:
            self.fail(f"Path planning failed with exception: {e}")


    def test_plot(self):
        """Test that the plot function runs without error"""
        obstacles = [Obstacle(center=(0, 0), radius=0.2)]
        rrt_star = RRTStar(start=(-0.8, -0.8), goal=(0.8, 0.8), bounds=[(-1, 1), (-1, 1)], obstacles=obstacles)
        path = rrt_star.plan()
        try:
            rrt_star.plot(path, show=False)
        except Exception as e:
            self.fail(f"Plotting failed with exception: {e}")
            
    def test_get_obstacle_info(self):
        """Test that the get_obstacle_info method returns correct obstacle info"""
        obstacles = [
            Obstacle(center=(0, 0), radius=0.5),
            Obstacle(center=(1, 1), radius=0.2),
            Obstacle(center=(-1, -1), radius=0.3)
        ]

        rrt_star = RRTStar(start=(-0.8, -0.8), goal=(0.8, 0.8), bounds=[(-1, 1), (-1, 1)], obstacles= obstacles)
        """Test that the get_obstacle_info method returns correct obstacle info"""
        expected_result = [
            {"center": [0, 0], "radius": 0.5},
            {"center": [1, 1], "radius": 0.2},
            {"center": [-1, -1], "radius": 0.3}
        ]
        obstacle_info = rrt_star.get_obstacle_info()
        self.assertEqual(obstacle_info, expected_result)

if __name__ == "__main__":
    unittest.main()
