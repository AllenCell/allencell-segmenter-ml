# Train a model from scratch

This workflow shows how to **train an ML model from scratch** using user's own data.

You can start training as soon as you have a {ref}`curation progress CSV saved<curation-csv>`

:::{note}
- Although computer with CPU only can run training, it takes significantly longer time than a computer with NVIDIA GPU
- Training may take an extended amount of time (e.g., overnight)
- training can run in the background as long as the application window (napari + the plugin) is open
:::


### STEPS

:::{figure} images/Tab_training_1.png
:::

1. `Curated image data source`: if you have completed curation using the plugin, this input field will be auto-populated

2. `Image channel`: this will also be auto-populated based on your training data

3. `Start from previous model`
    - `No`: allows training a model from scratch

4. `Patch size`: input the approximated dimension of your structure of interest

    - The input values must be multiples of 4 -- the fields will auto-correct to the closest value

5. `Model size`: this reflects the complexity of the model -- smaller model train faster while larger models train slower but may learn complex relationships better

6. `Number of epoch`: can start with a small value such as 10 to evaluate how quickly your computer can process each epoch

7. `Time out` (OPTIONAL): set up the model to stop training by a certain amount of time

8. Click `Start training`

    - A progress dialog will pop up to display the current progress and the current loss value
    - If a high value of epoch was entered, training may automatically stopped before it reaches the last epoch if the model can no long be improved

9. The plugin will notify you when the training is finished, together with the final loss value