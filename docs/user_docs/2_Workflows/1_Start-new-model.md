# Start a new model

In this workflow, you have access to all three modules: {bdg-dark-line}`Curation`, {bdg-dark-line}`Training`, and {bdg-dark-line}`Prediction`.

- curate a training dataset from their pool of available images (raw and segmenetation images)
- train a model from scratch or using the weight of an existing trained model (fine-tuning or iterative training)
- make segmentation prediction from the model trained in this workflow.

:::{figure} images/Workflow_new-model.png
:::

## STEPS

1. Select “Start a new model” option *AND* name your new model

    :::{tip}
    Name your model after the data that you will be using to train the model; e.g. LaminB1-interphase_1
    :::

2. Click `Apply`

3. The steps below have to be followed in sequence to ensure success:

    a. Start with {ref}`Curation<Use Curation Module>`

    b. Proceed to {ref}`"Training a model from scratch"<Train a model from scratch>` or {ref}`"Training an existing model"<Train a model iteratively>` once you have sufficient training dataset

    c. Run {ref}`Prediction & Thresholding<Use Prediction & Thresholding Modules>` only after you finish training your model


