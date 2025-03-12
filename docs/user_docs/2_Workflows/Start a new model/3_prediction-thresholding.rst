####
Prediction & Thresholding - rst
####

This module will generate both a probability map of the predicted segmentation and a binary segmentation image from a raw image


A. Run prediction
==================

This action generates a probablity map of the predicted segmentation from the raw image. 



a. Using on-screen image
------------------------

#. Load your raw image(s) by drag and drop them into the napari viewer

#. Select the correct channel of the raw image(s)

#. Select an output directory to store the probability map output 
  
#. Run prediction
  
#. The probability map output will be automatically saved as it's generated and displayed on-screen



b. Using an image directory
----------------------------

#. Select a directory

#. Select the correct channel of the raw images within the selected directory
  
#. Run prediction - a popup modal will display a progress bar as prediction runs through the directory

#. The probability map outputs will be saved in the background as they're generated

----


B. Thresholding
==================

This action converts the segmentation probability map to a binary image output

a. Using on-screen image
------------------------

#. Switch to the ``Thresholding`` tab

#. Select the probability map images generated from the previous step (Prediction)

#. Select a thresholding option and select appropriate value available within each option - the result will be generated live as you make adjustment

#. Once you're satisfied with a thresholding option/value, click to save your binary segmentation image



b. Using an image directory
----------------------------

#. Switch to the ``Thresholding`` tab

#. Select the directory of the probability map images generated from the previous step (Prediction)
  
#. Click to run thresholding on the entire directory - a popup modal will display a progress bar as prediction runs through the directory

#. The binary segmentation outputs will be saved in the background as they're generated

----


If you're satisfied with the model's performance, congratulations! You've successfully built a segmentation model tailored to your dataset. From now on, you can load this model throught the :ref:`"Select an existing model" workflow` and use it in your image analysis process. 
