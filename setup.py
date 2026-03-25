from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neural-spectral-codec",
    version="0.1.0",
    author="Kimun Park, Moon Gi Seok",
    author_email="",
    description="Neural Spectral Histogram Codec for LiDAR Loop Closing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DguAiCps/Neural-Spectral-Codec",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch==2.1.0",
        "torch-geometric==2.4.0",
        "torch-scatter==2.1.2",
        "torch-sparse==0.6.18",
        "numpy==1.24.3",
        "scipy==1.11.3",
        "open3d==0.18.0",
        "h5py==3.10.0",
        "pyyaml==6.0.1",
        "matplotlib==3.8.0",
        "seaborn==0.13.0",
        "wandb==0.16.0",
    ],
    extras_require={
        "dev": [
            "pytest==7.4.3",
            "pytest-cov==4.1.0",
        ],
    },
)
