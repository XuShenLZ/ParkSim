import setuptools

setuptools.setup(
    name="parksim",
    version="0.2.0",
    description="Parking Simulation",
    author="Xu Shen",
    author_email="xu_shen@berkeley.edu",
    packages=["parksim"],
    install_requires=[
        "pillow>=8.3.1",
        "numpy>=1.20.2",
        "scipy>=1.10.1",
        "tqdm>=4.60.0",
        "jupyterlab>=3.0.16",
        "opencv-python>=4.5.3.56",
        # "tensorboard>=2.5.0",
        "mosek>=9.3.6",
        "casadi~=3.5.5",
        "catkin_pkg",
        "empy",
        "lark",
        "dearpygui",
        "seaborn>=0.11.2",
        # "ray[tune]",
        # "pytorch_lightning",
        # "einops",
    ]
    # Dragon Lake Parking (DLP) Dataset should be built with its own source
    # Pytorch installation should follow the instructions on the website
)
