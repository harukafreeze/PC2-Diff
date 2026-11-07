from setuptools import setup, find_packages

setup(
    name="project_build",
    version="0.1.0",
    description="PC2Diff: Physical and Chemical Constraints for Structure-Based Molecular Generation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.23.0,<2.0.0",
        "scipy>=1.10.0",
        "rdkit>=2022.09.1",
        "pyyaml>=6.0",
        "easydict>=1.10",
        "lmdb>=1.4.0",
    ],
)
