# Frequently Asked Questions


## General

:::{card}
##### Why using the Allen Cell & Structure Segmenter ML plugin? 
^^^
Segmentation is often the first step in the image analysis process. However, achieving high-quality and accurate segmentation often requires significant time and effort to fine-tune the algorithms. Even with optimization, these algorithms might lack the flexibility to handle wide variablity in the datasets
  
  Meanwhile, machine learning has accelerated many high computing processes including image analysis. However, access to machine learning methods often requires specialized knowledge and coding expertise, creating a barrier for researchers whose expertise are outside of these fields.
  
:::


:::{card}
##### Is my progress saved if I exist the plugin or *napari*?
^^^
  - For curation: if you saved your progress using the plugin interface, a CSV file with list of images that you have already curated can be used to train a model
  - For training: there might be an intermediated/incompleted version of a model produced periodically during training saved, and you can use the weight of that version to train a newe model
  - For prediction: 
    - if prediction & thresholding were completed, the resulted probability-map segmentation image and thresholded binary segmentation image were wasautomatically saved in the designated output folder
    - if prediction was completed but thresholding was incompleted and not saved/applied, only the probability-map segmentation image was saved in the designated output folder
  
:::

<br>


## Interactions

:::{card}
##### Why do the images I loaded sometimes disappeared?
^^^
Currently our plugin will clear out any image on-screen if the user is switching from tab to tab. Please complete your interactions in the current tab before navigating away as your on-screen images may get cleared out

:::