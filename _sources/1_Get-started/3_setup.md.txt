# Setup

<br>

## 1. Start *napari*

Use the following command in your terminal or powershell:

```python
napari
```

You should see the below window pop up.

:::{figure} images/napari.png
:::

______________________________________________________________________

## 2. Start the Segmenter ML plugin

:::{figure} images/plugin-menu.png
:width: 500px
:align: left
:::

<br>

From napari menu, click on the Plugins tab and select “Allen Cell Segmenter - ML”

<br>

<br>

:::{note}
**The first time you run the plugin**, a pop-up window will appear prompting you to select a home directory for storing the plugin's related data.

Create a directory first then select it to be your **home directory**:
  - This directory will store your deep-learning models and curated dataset CSV
  - This directory persists if you reinstall or update your Allen Segmenter ML plugin
  - This directory can be changed later if needed - see {ref}`How do I...<Others>`
:::

______________________________________________________________________

## 3. Download pre-trained models (OPTIONAL)

Overall to start, we recommend testing our pre-trained models to see how they perform on your data.

The **pre-trained model download** option can be accessed through the `Help` dropdown at the upper right corner of the plugin window.

> Visit {ref}`Pre-trained models` to read more about the models we provided

:::{figure} images/download-models.png
:width: 500px

Download Models option from `Help` menu
:::

A popup window will appear and you can select which model you would like to download. Once the download is complete, another popup will let you know the download was successful and where the model was downloaded.

## 4. Select a model option to start

:::{figure} images/select-options.png
:width: 500px

Model options to select
:::

You are ready to start using the plugin!

Please select one of the workflows:

  > {ref}`A. "Select an existing model" workflow<Select an existing model>`
  >
  > {ref}`B. "Start a new model" workflow<Start a new model>`

:::{warning}
If no model option is selected, the `Curation`, `Training`, and `Prediction` tabs below this section will stay inactive and unavailable for interaction.
:::



