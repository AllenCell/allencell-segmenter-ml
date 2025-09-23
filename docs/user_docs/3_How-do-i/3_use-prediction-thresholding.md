# Use Prediction & Thresholding Modules

This module will generate both a {term}`probability map<Probability map image>` of the predicted segmentation and a binary segmentation image from a raw image.

## A. Run prediction

This action generates a probablity map of the predicted segmentation from a raw image.

:::{figure} images/Tab_prediction_1.png
:::

### a. If using on-screen image

1. Select the option "On-screen image(s)"
2. Load your raw images by drag and drop them into the *napari* viewer
3. Select valid raw images which has been populated under the "On-screen image(s)" label
4. Select a channel of the raw images
5. Select an output directory to store the probability map output
6. Click {bdg-primary}`Run` to run prediction
7. The probability map output will be automatically saved as it's generated and displayed on-screen

### b. If using an image directory

1. Select the option "Image directory"
2. Click {bdg-primary}`Browse` to select a directory
3. Select a channel of the raw images within the selected directory
4. Click {bdg-primary}`Run` to run prediction - a popup modal will display a progress bar as the plugin works through the directory
5. The probability map outputs will be saved in the background as they're generated

______________________________________________________________________

## B. Thresholding

This action converts the segmentation probability map to a binary image output. 

**Threshold usage**

We have validated our own models (e.g. MegaSeg) using a 50% threshold i.e., pixels for which the model output is lower than 128 will be classified as background.

This is because model prediction can be seen as a probability map where each pixel has an associated probability of it being foreground or background. A probability threshold of 50% represents, any pixel with higher associated probability will be part of the foreground or the target structure. Pixels with less than 50% associated probability will be part of the background.

However, we encourage users to explore different threshold values using the threshold functionality of the plugin as for some applications it will be more appropriate to only segment the bright – higher probability regions. While in some cases a lower threshold will be much more meaningful to include the dim regions also at the cost of maybe over segmenting bright regions.

:::{figure} images/Tab_thresholding.png
:alt: view of thresholding tab of Segmenter ML plugin in napari
:::

### a. If using on-screen image

1. The plugin will auto-switch to the {bdg-secondary}`Thresholding` tab after the prediction is completed
    :::{caution}
    Switching back to {bdg-secondary}`Prediction` will clear out current on-screen images, please finish your interactions on this module before switching. 
    :::
2. Select the option "On-screen image(s)"
3. Select the newly generated probability map images populated under the "On-screen image(s)" label
4. Select an output directory to store the thresholded output
5. Select a thresholding option and select appropriate value available within each option - the thresholded result images will be updated in real time as you're making adjustments
6. Once you're satisfied with a thresholding option/value, click {bdg-primary}`Apply and Save` to save your thresholded images

### b. If using an image directory

1. The plugin will auto-switch to the {bdg-secondary}`Thresholding` tab after the prediction is completed
2. Select the option "Image directory"
3. Select the **subdirectory** `Seg` created by the plugin within the probability map output directory you've selected in the previous step
4. Select an output directory to store the thresholded output 
5. Select a thresholding option and select appropriate value available within each option
6. Click {bdg-primary}`Apply and Save` to run thresholding - a popup modal will display a progress bar as the plugin works through the directory
7. The thresholded binary images will be saved in the background as they're generated
8. To review the generated thresholded images, drag and drop them into the *napari* viewer
    - In *napari*'s Layer Control panel, adjust the "Contrast limits" range from 0-255 to 0-1 by sliding the right handle all the way to the left to correctly view the image
        :::{figure} images/Thresholded-images_contrast-limits.png
        :alt: adjust contrast limits of the thresholded result image to correctly view the image
        :::

        :::{tip}
        Right-click on the slider bar to show the detailed view of the slider
        :::

:::{caution}
- If the signal-to-noise ratio in a segmentation result is low, the thresholding output might be empty.
:::

______________________________________________________________________

## C. Next steps

If you're satisfied with the performance of a model you've trained, congratulations! You've successfully built a segmentation model tailored to your dataset. From now on, you can load this model throught the {ref}`"Select an existing model" workflow` and use it in your image analysis process.

If you are not satisfied with the model's performance, there multiple ways for the next steps:

- Improve your training data:
    - add more high-quality images
    - improve the quality of the raw images and segmentations
    - re-curate your training dataset, be more thorough with removing data that might affect the model's performance
- Continue training your model:
    - Start a new model and select the weight of the previous model you've used or trained ({ref}`workflow here<Train a model iteratively>`)