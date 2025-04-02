from setuptools import setup, find_packages

setup(
    name="conditional_diffusion_motion",
    version="0.1.0",
    description="A package to simulate diffusion motion with conditional diffusion",
    author="Arthur Haffemayer",
    author_email="arthur.haffemayer@laas.fr",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)