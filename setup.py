import setuptools

setuptools.setup(
    name="parksim", 
    version="0.1.1",
    description='Parking Simulation',
    author='Xu Shen',
    author_email='xu_shen@berkeley.edu',
    packages=[
        'intent_predict',
        'spot_detector'
    ],
    install_requires=[
        "pillow==8.3.1",
        "numpy==1.20.2",
        "tqdm==4.60.0",
        "jupyterlab==3.0.16",
        "opencv-python==4.5.3.56"
    ]
    # Dragon Lake Parking (DLP) Dataset should be built with its own source
    # Pytorch installation should follow the instructions on the website
)