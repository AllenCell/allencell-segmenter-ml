# Use Prediction & Thresholding Modules

This module will generate both a {term}`probability map<Probability map image>` of the predicted segmentation and a binary segmentation image from a raw image.

## A. Run prediction

:::{figure} images/Tab_prediction_1.png
{bdg-danger}`need updated version`
:::

This action generates a probablity map of the predicted segmentation from the raw image.

### a. Using on-screen image

1. Load your raw image(s) by drag and drop them into the napari viewer
2. Select the correct channel of the raw image(s)
3. Select an output directory to store the probability map output
4. Run prediction
5. The probability map output will be automatically saved as it's generated and displayed on-screen

### b. Using an image directory

1. Select a directory
2. Select the correct channel of the raw images within the selected directory
3. Run prediction - a popup modal will display a progress bar as prediction runs through the directory
4. The probability map outputs will be saved in the background as they're generated

:::{caution} 
Cancelling a prediction run might take several minutes
:::
______________________________________________________________________

## B. Thresholding

This action converts the segmentation probability map to a binary image output. 

Threshold usage:
We have validated our own models (e.g. MegaSeg) using a 50% threshold i.e., pixels for which the model output is lower than 128 will be classified as background.

This is because model prediction can be seen as a probability map where each pixel has an associated probability of it being foreground or background. A probability threshold of 50% represents, any pixel with higher associated probability will be part of the foreground or the target structure. Pixels with less than 50% associated probability will be part of the background.

However, we encourage users to explore different threshold values using the threshold functionality of the plugin as for some applications it will be more appropriate to only segment the bright – higher probability regions. While in some cases a lower threshold will be much more meaningful to include the dim regions also at the cost of maybe over segmenting bright regions.

### a. Using on-screen image

1. Switch to the `Thresholding` tab
2. Select the probability map images generated from the previous step (Prediction)
3. Select a thresholding option and select appropriate value available within each option - the result will be generated live as you make adjustment
4. Once you're satisfied with a thresholding option/value, click to save your binary segmentation image

### b. Using an image directory

1. Switch to the `Thresholding` tab
2. Select the directory of the probability map images generated from the previous step (Prediction)
3. Click to run thresholding on the entire directory - a popup modal will display a progress bar as prediction runs through the directory
4. The binary segmentation outputs will be saved in the background as they're generated

:::{warning}
If the signal-to-noise ratio in a segmentation result is low, the thresholding output might be empty. 
:::

______________________________________________________________________

## C. Next steps

If using the model you trained and you're satisfied with the model's performance, congratulations! You've successfully built a segmentation model tailored to your dataset. From now on, you can load this model throught the {ref}`"Select an existing model" workflow` and use it in your image analysis process.

If you are not satisfied with the model's performance, there multiple ways for the next steps:

- Improve your training data:
    - add more high-quality images
    - improve the quality of the raw images and segmentations
    - re-curate your training dataset, be more thorough with removing data that might affect the model's performance
- Continue training your model:
    - Start a new model and select the weight of the previous model you've used or trained ({ref}`workflow here<Train a model iteratively>`)