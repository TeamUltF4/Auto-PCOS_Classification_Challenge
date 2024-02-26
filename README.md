# Pipeline Overview

Our pipeline involves the development of a Convolutional Neural Network (CNN) model for automatic classification of healthy and unhealthy frames in ultrasound imaging. The pipeline consists of the following stages:

1. Data Preprocessing: Preprocessing of ultrasound images to enhance quality and remove noise.
2. Model Training: Training an improved CNN model on the preprocessed images.
3. Model Evaluation: Evaluating the trained model on validation and testing datasets to assess its classification performance.
4. Interpretability Analysis: Analyzing interpretability plots to understand the model's decision-making process.

## Achieved Results on Validation Dataset

### Evaluation Metrics

| Metric             | Value   |
|--------------------|---------|
| Accuracy           | 0.85    |
| Precision          | 0.82    |
| Recall             | 0.88    |
| F1 Score           | 0.85    |

### Best Frames Classification

Below are pictures of the best frames selected from the validation dataset showing their classification:

1. Frame 1 ![Frame 1](validation_frame1.jpg)
   - Interpretability Plot ![Interpretability Plot 1](validation_interpretability_plot1.jpg)
   
2. Frame 2 ![Frame 2](validation_frame2.jpg)
   - Interpretability Plot ![Interpretability Plot 2](validation_interpretability_plot2.jpg)
   
3. Frame 3 ![Frame 3](validation_frame3.jpg)
   - Interpretability Plot ![Interpretability Plot 3](validation_interpretability_plot3.jpg)
   
4. Frame 4 ![Frame 4](validation_frame4.jpg)
   - Interpretability Plot ![Interpretability Plot 4](validation_interpretability_plot4.jpg)
   
5. Frame 5 ![Frame 5](validation_frame5.jpg)
   - Interpretability Plot ![Interpretability Plot 5](validation_interpretability_plot5.jpg)

## Achieved Results on Testing Dataset

### Threshold Evaluation Metrics

Here are the evaluation metrics for different thresholds on the testing dataset:

| Threshold | Mean Average Precision (mAP) | Accuracy | F1 Score |
|-----------|------------------------------|----------|----------|
| 0.1       | 0.9749                       | 0.9789   | 0.9873   |
| 0.2       | 0.9828                       | 0.9857   | 0.9913   |
| 0.3       | 0.9869                       | 0.9891   | 0.9934   |
| 0.4       | 0.9893                       | 0.9911   | 0.9946   |
| 0.5       | 1.0000                       | 1.0000   | 1.0000   |
| 0.6       | 0.9941                       | 0.9734   | 0.9835   |
| 0.7       | 0.9704                       | 0.8658   | 0.9108   |
| 0.8       | 0.9599                       | 0.8181   | 0.8752   |
| 0.9       | 0.9518                       | 0.7813   | 0.8460   |

### Best Frames Classification

Below are pictures of the best frames selected from the testing dataset showing their classification:

1. Frame 1 ![Frame 1](testing_frame1.jpg)
   - Interpretability Plot ![Interpretability Plot 1](testing_interpretability_plot1.jpg)
   
2. Frame 2 ![Frame 2](testing_frame2.jpg)
   - Interpretability Plot ![Interpretability Plot 2](testing_interpretability_plot2.jpg)
   
3. Frame 3 ![Frame 3](testing_frame3.jpg)
   - Interpretability Plot ![Interpretability Plot 3](testing_interpretability_plot3.jpg)
   
4. Frame 4 ![Frame 4](testing_frame4.jpg)
   - Interpretability Plot ![Interpretability Plot 4](testing_interpretability_plot4.jpg)
   
5. Frame 5 ![Frame 5](testing_frame5.jpg)
   - Interpretability Plot ![Interpretability Plot 5](testing_interpretability_plot5.jpg)

