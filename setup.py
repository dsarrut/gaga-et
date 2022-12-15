import setuptools
from setuptools import find_packages

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gaga_et",
    version="0.1",
    author="David Sarrut",
    author_email="david.sarrut@creatis.insa-lyon.fr",
    description="Tools for emission tomography with GATE: PET and SPECT GAN simulations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dsarrut/gagapet",
    packages=find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        "opengate",
        # 'torch'   # better to install torch manually to match cuda version
    ],
    scripts=[
        "bin/pet_gaga_training_dataset",
        "bin/pet_ideal_reconstruction",
        "bin/image_profile",
        "bin/spect_arf_training_dataset",
        "bin/spect_gaga_training_dataset",
        "bin/spect_simulation",
        "bin/spect_generate_img"
    ],
)
