# Morphology-Feature-Extraction

This code extracts morphological features of sperm cells under with the microscopic system we used, although different paramters of functions can be inserted to make it suit other systems.

The algorithm filters out images of cells that do not allign with the expected, in focus, noiseless image of a sperm cell whose head is parallel to the slide.

The algorithm also rotates the cells in a specific orientation that facilitates the feature extraction proccess and can also be used to help neural networks learn on these images.

getting_betas.py is somewhat irrelevant for future users of this code, as it is just the function of retrieving the maximum raw OPD value for the image, in order to turn the image to it's original OPD values.
