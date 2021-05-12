# Time-series-Metric-Learning
In this project work the goal was to experiment with Deep Metric Learning techniques, in particular the Triplet Loss, applied in the context of a multivariate time series classification problem. In literature, techniques of this type have been successfully applied in the context of images, especially in the contexts of Face Verification, Identification and Image Retrieval. We wanted to experiment how these techniques works in the time series context. 
In this repository there are 5 colab notebooks:
 - **Data visualization.ipynb**: in this notebook it possible to visualize the time-series and the distribution of the classes in the dataset
 - **Baseline CrossEntropy.ipynb**: as baseline for our experiments, the representation created by the last layer of the model was used considering the CrossEntropy as loss
 - **Triplet Loss all-triplet.ipynb**: implementation of the Triplet Loss with all triplet mining strategy 
 - **Triplet Loss semi-hard.ipynb**: implementation of the Triplet Loss with semi-hard mining strategy 
 - **Triplet Loss all-triplet + data augmentation.ipynb**: implementation of the Triplet Loss with all triplet mining strategy and a oversampling tecnique to balance the number of examples per class

**NB: Extract the dataset "project_dataset.tar.xz" in the folder "dataset" before running the experiments**



## Baseline
![alt text](https://github.com/andreafuschino/Generalized-Hough-Transform/blob/main/output.png)
## Triplet Loss all-triplet
![alt text](https://github.com/andreafuschino/Generalized-Hough-Transform/blob/main/output.png)
## Triplet Loss semi-hard
![alt text](https://github.com/andreafuschino/Generalized-Hough-Transform/blob/main/output.png)
## Triplet Loss all-triplet + data augmentation
![alt text](https://github.com/andreafuschino/Generalized-Hough-Transform/blob/main/output.png)
