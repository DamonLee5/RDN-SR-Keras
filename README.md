# course-project-DamonLee5
# Acknowledgement
For the best of my knowledge, my project is the first RDN implementation written by Keras. However, I still need to acknowledge that https://github.com/yulunzhang/RDN and https://github.com/hengchuan/RDN-TensorFlow help me for some hyperparameter setting.

# Data preparation
## utils.py
This file generates our training datasets for three different degradations, including 5 datasets.
  - Bicubic degradation with factor 2, 3, and 4.
  - BD degradation with factor 3
  - DN degradation with factor 3
Also this file generates 25 datasets for 5 degradations on 5 benchmark datasets, which have not been seen while training.
To generate those dataset, I wrote function to chop, rescale, filter, read, and save the image.

# DNN Model
I implement this project by Keras. I utilize a standard Keras framework with a data_loader and model file.
## data_loader.py
This file is used for load the data from the dataset for training, sampling and testing.
## model.py
This file include the entire RDN network. Also, it can test on validation set while traing and load a trained model for testing.

# Training
RDB_run.sh is used to train in cluster by qsub.

# Test Result Generation
## test.py
Since we need to calculate the metric, and online calculate takes too long, I first generate the entire reconstructed high resolution dataset for metric calculation. This result dataset include ground truth and reconstructed image.

## Metric.py
Calculate the Metric among all 25 testing datasets. Result is in the Metric.log, which is consistent with the result displayed in the term paper.

## loss_plot.ipynb
Plot loss from training log file.

## Utils_test.ipynb
Generate an entire high resolution image from reconstructed patches.

# Saved model
https://drive.google.com/open?id=1sQZO1ZZ4MIYM_vF5wOm6e9F8OiyeoF_H
- RDB_best_16.h5, BI degradation, factor 3
- RDB_best_17.h5, BI degradation, factor 2
- RDB_best_18.h5, BI degradation, factor 4
- RDB_best_19.h5, BD degradation, factor 3
- RDB_best_20.h5, DN degradation, factor 3
