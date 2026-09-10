from setuptools import setup, find_packages

setup(
    name="project_build",
    version="0.1.0",
    description="Project BUILD: Physics-Guided Equivariant Diffusion for 3D Drug Design",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "numpy<2.0.0",
        "scipy>=1.10.0",
        "rdkit",
        "pyyaml>=6.0",
        "easydict>=1.10",
    ],
)
