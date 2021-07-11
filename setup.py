import setuptools

setuptools.setup(
    name="parksim", 
    version="0.1.0",
    description='Parking Simulation',
    author='Xu Shen',
    author_email='xu_shen@berkeley.edu',
    packages=['intent_predict'],
    install_requires=[
    "numpy==1.20.2",
    "tqdm==4.60.0",
    "ipykernel==6.0.1",
    "jupyterlab==3.0.16"
    ]
    # Dragon Lake Parking (DLP) Dataset should be built with its own source
    # Pytorch installation should follow the instructions on the website
)