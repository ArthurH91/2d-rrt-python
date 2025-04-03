from setuptools import setup, find_packages

setup(
    name="rrt_planner_python",
    version="0.1.0",
    description="A package to create 2D RRT (Rapidly-exploring Random Tree) paths.",
    author="Arthur Haffemayer",
    author_email="arthur.haffemayer@laas.fr",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)