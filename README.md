ONC
==============================

Welcome to the Endeavour Analysis repository. Here you can find a script to generate
a daily GIF of earthquakes at the Endeavour Segment from location.mat files made with
the Endeavour Autolocate software developed by Krauss and Wilcock (University of Wash.).

Note that this has only been tested in a Mac environment. The script must be run from the
src directory and is dependent on ImageMagick which can be downloaded using HomeBrew
(brew install ImageMagick).

For running the script, you should only need to adjust the mat_file and input
parameters, which are described with the script.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── input          <- Input files to the program, immutable.
    │   └── raw            <- Location files from the Endeavour Autolocate software
    │
    ├── results            <- Generated analysis
    │   └── images         <- Generated graphics and figures
    │
    ├── requirements.txt   <- Requirements for reproducing the analysis environment
    │
    └── src                <- Source code for use in this project.

--------
