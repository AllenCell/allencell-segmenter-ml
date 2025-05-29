
# Use Curation Module



## 1. Data preparation

Organize your image data into directories as the following:

- Raw image directory
  - Contains your original microscopy image data
  - Accepts: `.tiff`, `.ome-tiff`, `.czi`

- Segmentation 1 directory
  - Contains your segmentation of the structure of interest
  - Accepts: `.tiff`, `.ome-tiff`, `.czi` 

- Segmentation 2 directory (optional)
  - Contains *optional* complementary segmentation image data, which could be generated from the same raw image data for the same structure but using different segmentation algorithms.
  - It is useful when Seg 1 fails predictably in certain cases (e.g. a segmentation that works during mitosis to supplement an interphase segmentation)
  - Accepts: `.tiff`, `.ome-tiff`, `.czi`

**Other requirements**
- At least **4 raw images** and **4 corresponding segmentation files** are required. Using a larger number of images is strongly recommended to train a robust and accurate segmentation model.
- The number of files should be consistent across directories
- It's important to have your raw & segmentations named appropriately for the plugin to display them correctly. For example:
  - `AICS-13_1516_raw.tiff` (raw)
  - `AICS-13_1516_segmentation_interphase.tiff` (seg 1)
  - `AICS-13_1516_segmentation_mitotic.tiff` (seg 2)



______________________________________________________________________

## 2. Select input data

After applying the "Start a new model" option with an appropriate model name, the plugin will display the Curation tab as shown below:

:::{figure} images/Tab_curation_s1.png
:alt: screen 1 of curation tab of Segmenter ML plugin in *napari*
:align: left
:::

**STEPS**

1. **Directory** -- select the directories you have prepared

2. **Image channel** -- the image channel dropdown will display available channels detected in the directories you have loaded. Channel often starts at `0`

3. Click {bdg-primary}`Start`

______________________________________________________________________

## 3. Start curation

**VIEWPORT**
:::{figure} images/Image-layers.png
:alt: images layer in Layer List panel
:::

In the viewport, your images will be displayed one set at a time, with the segmentations overlaid on top of the raw image in the following order top to bottom:
  - **`Seg 2`** (if available): `label` layer: teal color
  - **`Seg 1`**: `label` layer: orange color
  - **`Raw`**: `image` layer: grayscale color

**PLUGIN INTERFACE**

:::{figure} images/Tab_curation_s2.png
:alt: screen 2 of curation tab
:::

<br>

### a. Sorting

- Revisit the concept of [sorting](sorting-concept) if needed

(curation-csv)=

**STEPS**

:::{figure} images/Sorting_ui.png
:alt: sorting interface
:::

1. **User this image for training?**
    - Select {bdg-dark}`Yes` (the default selection) to use the image pair for training
    - Select {bdg-dark}`No` to skip this image pair
    
2. **Select a {term}`base segmentation`**
    -  `Seg 1` is selected by default
    - if `Seg 2` is available, you can choose to use either `Seg 1` or `Seg 2`

3. Click {bdg-primary}`Next` to proceed to the next image pair if no further action needed (e.g., excluding or merging), 
    - The progress bar displays your current progress, and show how many image pairs you have reviewed
      :::{figure} images/Curation-progress_ui.png
      :alt: Curation progress section
      :::
    - When at least 4 image pairs have been selected to use for training, the {bdg-primary}`Save curation progress` button will be enabled for you to save your progress
      :::{figure} images/Curation-progress_save.png
      :alt: Save curation progress
      :::

<br>

### b. Create an excluding mask (OPTIONAL)

- Revisit the concept of [excluding](excluding-concept) if needed

**STEPS**

1\. Click the {bdg-primary}`Create` button under the **Excluding mask** section&#10;
:::{figure} images/Mask_excluding_ui.png
:alt: create an excluding mask
:::

2\. Your cursor automatically changes to the polygon drawing tool (cross-hair icon over the viewport)&#10;

3\. Start drawing shapes to cover areas of the image you want to exclude, double-click to close the shape, and review in z using the slider at the bottom of the viewport to ensure the shapes are still appropriate
:::{figure} images/Mask_excluding_draw.png
:alt: draw excluding mask
::: 
:::{figure} images/z-slider.png
:alt: slider to navigate through z slices
:::
:::{tip}
  Precision isn't critical when drawing, as long as the subpar segmentation areas are adequately covered. We aim to make this process quick, especially since there may be a large number of images for you to review.
:::

4\. When you're finished with all shapes, click {bdg-primary}`Save` to save your excluding mask&#10;

5\. Alternatively, if you:
- no longer need an excluding mask, click {bdg-primary}`Delete`
- wish to start over, click {bdg-primary}`Delete` then click {bdg-primary}`Create` to add a new excluding mask&#10;

:::{note}
  Each image pair will have only one excluding mask, clicking {bdg-primary}`Save` multiple times will resave the same mask or clicking {bdg-primary}`Create` multiple times will overwrite the previously created mask
:::

<br>

(merging)=

### c. Create a merging mask (OPTIONAL)

- Revisit the concept of [merging](merging-concept) if needed

- This function is **only** enabled if `Seg 2` is also present in additional to `Seg 1`

**STEPS** (similar to the Excluding mask's steps)

1\. Click the {bdg-primary}`Create` button under the **Merging mask** section&#10;
:::{figure} images/Mask_merging_ui.png
:alt: create merging mask
:::

2\. Your cursor automatically changes to the polygon drawing tool&#10;

3\. Start drawing shapes to cover areas of the image you want to be overwritten, double-click to close the shape, and review in z using the slider at the bottom of the viewport to ensure the shapes are still appropriate
:::{figure} images/Mask_merging_draw.png
:alt: draw merging mask
:::
:::{figure} images/z-slider.png
:alt: slider to navigate through z slices
:::

4\. When you're finished with all shapes, click {bdg-primary}`Save` to save your merging mask&#10;

5\. Alternatively, if you:
- no longer need a merging mask, click {bdg-primary}`Delete`
- wish to start over, click {bdg-primary}`Delete` then click {bdg-primary}`Create` to add a new merging mask&#10;

:::{note}
Each image pair will have only one merging mask, clicking {bdg-primary}`Save` multiple times will resave the same mask or clicking {bdg-primary}`Create` multiple times will overwrite the previously created mask
:::

:::{warning}
Merging mask and excluding mask can't overlap each other. 
:::

______________________________________________________________________

## 4. Complete curation

Upon reaching the last image indicated in the progress bar

- Click {bdg-primary}`Finish` next to the progress bar
:::{figure} images/Curation_complete.png
:alt: click finish to complete curation
:::
- Your curation progress will be saved automatically
- A dialog will prompt you to switch to the {bdg-dark}`Training` tab to start training
:::{figure} images/Curation_complete_dialog.png
:alt: curation complete dialog box
:width: 700px
:align: left
::: <br>

&#10;

:::{caution}
- If you close the plugin **while in the middle of the curation process**, the plugin will not save your progress.
- Your curation progress will only be saved once you have finished curating your dataset or have clicked {bdg-primary}`Save curation progress`. 
- Curation progress is saved to a CSV and can be used for training immediately once saved, but currently it cannot be used to resume curation if you exit the curation process.
:::