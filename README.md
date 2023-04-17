Multitarget Estimation, Tracking, and Information Synthesis (METIS)


Much of the code for coordinate frame rotations and numerical integration is adapted from a MATLAB library by Ben K. Bradley.



Install Instructions:
Open a terminal window or anaconda prompt and navigate to the directory where you have cloned this repo.  Perform the following steps to update conda and create a virutal environment called metis.

$ conda update conda

$ conda env create -f environment_metis.yaml

Next, activate the new virtual environment and install sgp4 (not included in the .yaml at this time).

$ conda activate metis

$ conda install -c conda-forge sgp4

