# Glossary

Terms are organized alphabetically

:::{glossary}
Batch
    Number of examples seen per model update step
Cost Function/Loss Function
    A lost function is used to calculate the cost, which is the difference between the predicted value and the actual value
Epochs
    An epoch is one complete pass through the entire training dataset. More epochs yields better performance at the cost of training time
Excluding mask
    Indicates areas of the image set (raw, seg(s)) that will be excluded from training
Image dimension
    Dimensionality of your image data, the plugin supports 2D and 3D data
Inference or Prediction
    Uses a trained model to make prediction on new data
Learning rate
    Scales the magnitude of model weight updates
Loss value
    Error between model prediction and ground truth
Merging mask
    User-drawn custom shapes, indicateing areas of the base segmentation that will be overwritten by the other segmentation
Model size
    Defines the complexity of the model - smaller models train faster, while large models train slower but may learn complex relationships better
Model weight
    The parameters learned by the model during training
Patch size
    Patch size to split images into during training. Should encompass the structure of interest and all dimensions should be evenly divisble by 4. If 2D, Z can be left blank. Larger patch sizes will take up more memory and have slower training. 
Probability map image
    Raw model prediction, showing the probability of each voxel belonging to the structure of interest versus background
Raw image
    Original microscopy images (.czi, .ome.tiff, .tiff)
Seg 1
    Segmentation of the structure of interest (.czi, .ome.tiff, .tiff) 
Seg 2 (optional)
    (Optional) Complementary segmentation, useful if Seg 1 fails predictably (e.g. a segmentation that works during mitosis to supplement an interphase segmentation)
Segmentation model
    A type of ML model that separates the structures of interest from their background in a 2D/3D mircroscopy image
Threshold_otsu
    A method for automatically selecting a threshold for binarization. For more information, please see https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu
Thresholded binary image
    The result from thresholding a probability-map image
Thresholding value
    The probability above which a pixel is considered in the foreground
Weights/ Bias
    The learnable parameters in a model
:::