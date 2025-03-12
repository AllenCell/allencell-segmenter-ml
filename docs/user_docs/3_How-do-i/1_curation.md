# Use Curation Module

## 1. Data preparation

Organize your image data into directories as the following:

- General requirements
  : - At least **4 raw images** and **4 corresponding segmentation files** are required. However, using a larger number of images is strongly recommended to train a robust and accurate segmentation model.



    - File names (excluding suffixes) and the number of files should be consistent across the following directories to ensure the plugin loads them correctly. See examples below.



    - These directories do not need to be in the same directory as your home directory



    - Although **not recommended**, both raw image & segmentation can be stored as channels within the same multi-channel file
- Raw
  : This directory contains your original microscopy image data

    Accepts: `.tiff`, `.ome-tiff`, `.czi`

    Example: `image-1_raw.tiff`
- Seg 1
  : This directory contains your segmentation of the structure of interest

    Accepts: `.tiff`, `.ome-tiff`, `.czi`

    Examples: `image-1_seg-1.tiff`, `image-1_interface.tiff`, etc.
- Seg 2 (optional)
  : This directory contains *optional* complementary segmentation image data, which could have been generated from the same raw image data for the same structure, but using different segmentation algorithms. It is useful when Seg 1 fails predictably in certain cases (e.g. a segmentation that works during mitosis to supplement an interphase segmentation)

    Accepts: `.tiff`, `.ome-tiff`, `.czi`

    Examples: `image-1_seg-2.tiff`, `image-1_mitotic.tiff`, etc.

______________________________________________________________________

## 2. Select input data

:::{figure} images/curation_main-screen.png
:width: 50%
:align: left
:::

1. Select the directories you have prepared&#10;

2. The image channel dropdown will display available channels detected in the directories you have loaded. Please select the appropriate channel. Channel often starts at `0`&#10;

3. Click `Start`&#10;

<br>
______________________________________________________________________

## 3. Start curation

:::{figure} images/Tab_curation.png
:::

:::{warning}
- If you close the plugin **while in the middle** of the curation process, please note that the plugin does not save your progress automatically. Your curation progress will only be saved once you have finished curating your dataset or clicked `Save curation progress`.
- Curation progress is saved to a CSV in the background, which can be used for training immediately if needed, but it cannot be used to resume curation if you accidentally exist the curation process
:::

### a. Sorting

:::{figure} images/Curation_sorting.png
:::

Review existing pool of raw images and corresponding segmentations, then select only the high-quality images to be used as training data

In the viewport, your images will be displayed one pair at a time, with the segmentations overlaid on top of the raw image in the following order top to bottom:
  - `Seg 2` (if available): `label` layer: teal color
  - `Seg 1`: `label` layer: orange color
  - `Raw`: `image` layer: grayscale color

(curation-csv)=

**STEPS**

1. Select `Yes` (the default selection) to use the image pair for training or `No` to skip this image pair

2. Select **base segmentation**: 
    - **Base segmentation**: the primary segmentation used for training, this will also play a role when we get to the {ref}`Merging<merging>` function
    - the default value is `Seg 1`
    - if `Seg 2` is available, you can choose to use either `Seg 1` (the default selection) or `Seg 2`
    

3. If no further action needed (e.g., excluding low-quality regions or overwriting regions), click `Next` to proceed to the next image pair
    - The progress bar displays your current progress, and show how many image pairs you have reviewed

4. If at least 4 image pairs have been selected for use, the `Save curation progress` button will be available for you to save your progress

### b. Excluding: Create an excluding mask (OPTIONAL)

**DEFINITION**

- An excluding mask contains region(s) of an image to be excluded from the training process

- The regions to be excluded typically are subpar segmentation that could negatively affect the model's performance

- The regions are drawn by using *napari*'s polygon drawing tool

- For 3D images, the regions are drawn as 2D xy shapes, but the excluding effect will be propagated through z with the same xy shapes

   :::{tip}
  Precision isn't critical when drawing, as long as the subpar segmentation areas are adequately covered. We aim to make this process quick, especially since there may be a large number of images for you to review.
  :::

**STEPS**

1\. Click the `Create` button under the **Excluding mask** section&#10;

2\. Your cursor automatically changes to the polygon tool&#10;

3\. Start drawing shapes to cover areas of the image you want to exclude, and review the z-slices to ensure the shapes are still appropriate. When you're finished with each shape, **double-click** to close it.

4\. When you're finished with all shapes, click `Save` to save your excluding mask&#10;

5\. Alternatively, if you no longer need an excluding mask or wish to start a mask over, click `Delete`, then click `Create` to add a new mask&#10;
    
  :::{note}
  Each image pair will have only one excluding mask, clicking `Save` multiple times will resave the same mask or clicking `Create` multiple times will overwrite the previously created mask
  :::

(merging)=

### c. Merging (overwriting): Create a merging mask (OPTIONAL)

**DEFINITION**

- This function is **only** available if `Seg 2` is also present in additional to `Seg 1`

- You can choose to use either `Seg 1` (the default selection) or `Seg 2` as the **base segmentation**
- The other segmentation will serve as the **complementary segmentation**, which can then be used for the merging function

- A merging mask contains regions of the base segmentation to be overwritten by the complementary segmentation
- This is useful when the base segmentation is accurate for most of the image but fails in certain areas, particularly where the morphology of the structure of interest differs significantly from the common morphology. In such cases, the complementary segmentation, which was generated specifically to target these areas, can be used instead. This function effectively combines the two segmentations into a single ground-truth segmentation for training.

**STEPS** (this is similar to the **Excluding mask**'s steps)

1\. Click the `Create` button under the **Merging mask** section&#10;

2\. Your cursor automatically changes to the polygon tool&#10;

3\. Start drawing shapes to cover areas of the image you want to be overwritten, and review the z-slices to ensure the shapes are still appropriate. When you're finished with each shape, **double-click** to close it.&#10;

4\. When you're finished with all shapes, click `Save` to save your merging mask&#10;

5\. Alternatively, if you no longer need an merging mask or wish to start a mask over, click `Delete`, then click `Create` to add a new mask&#10;

:::{note}
Each image pair will have only one merging mask, clicking `Save` multiple times will resave the same mask or clicking `Create` multiple times will overwrite the previously created mask
:::

:::{warning}
Merging mask and excluding mask can't overlap each other. 
:::

______________________________________________________________________

## 4. Curation Complete

Upon the comletion of the curation process:

- Your curation progress will be saved automatically
- A dialog will prompt you to switch to the `Training` tab to start training
