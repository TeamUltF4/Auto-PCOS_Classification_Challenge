# Pipeline Overview

Our pipeline involves the development of a Convolutional Neural Network (CNN) model for automatic classification of healthy and unhealthy frames in ultrasound imaging. The pipeline consists of the following stages:

1. Data Preprocessing: Preprocessing of ultrasound images to enhance quality and remove noise.
2. Model Training: Training an improved CNN model on the preprocessed images.
3. Model Evaluation: Evaluating the trained model on validation and testing datasets to assess its classification performance.
4. Interpretability Analysis: Analyzing interpretability plots to understand the model's decision-making process.

## Model Architecture

```python
# Section 7: Build Improved CNN Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
```

## Achieved Results on Validation Dataset

### Evaluation Metrics

| Metric             | Value   |
|--------------------|---------|
| Accuracy           | 0.85    |
| Precision          | 0.82    |
| Recall             | 0.88    |
| F1 Score           | 0.85    |

### Best Frames Classification

Below are pictures of the best frames selected from the validating dataset showing their classification (all healthy dataset):

| Frame 1 | Frame 2 | Frame 3 | Frame 4 | Frame 5 |
|---------|---------|---------|---------|---------|
| [![Frame 1](https://drive.google.com/uc?id=10zqCqFGiJIzfVevZH5OnEEooplpUXZBR)](https://drive.google.com/file/d/10zqCqFGiJIzfVevZH5OnEEooplpUXZBR/view?usp=sharing) | [![Frame 2](https://drive.google.com/uc?id=14L_E-7iw4czYvJ98sm93uWa9M22JXhPS)](https://drive.google.com/file/d/14L_E-7iw4czYvJ98sm93uWa9M22JXhPS/view?usp=sharing) | [![Frame 3](https://drive.google.com/uc?id=1uZRgucjDWWD0mXnYMc_Zbmh9Nh5SiLM1)](https://drive.google.com/file/d/1uZRgucjDWWD0mXnYMc_Zbmh9Nh5SiLM1/view?usp=sharing) | [![Frame 4](https://drive.google.com/uc?id=1RFfw3HJ6vTQyFIu7SmmB-FV6MK4B6Q8c)](https://drive.google.com/file/d/1RFfw3HJ6vTQyFIu7SmmB-FV6MK4B6Q8c/view?usp=sharing) | [![Frame 5](https://drive.google.com/uc?id=19JTGbISyAbfQJXXpu_geFLofiXNrfaLT)](https://drive.google.com/file/d/19JTGbISyAbfQJXXpu_geFLofiXNrfaLT/view?usp=sharing) |

### Interpretability Plots

| Interpretability Plot 1 | Interpretability Plot 2 | Interpretability Plot 3 | Interpretability Plot 4 | Interpretability Plot 5 |
|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| [![Interpretability Plot 1](https://drive.google.com/uc?id=17GOj0Mq3E3BmXoEwHaEMVWm_6vmbxC0t)](https://drive.google.com/file/d/17GOj0Mq3E3BmXoEwHaEMVWm_6vmbxC0t/view?usp=sharing) | [![Interpretability Plot 2](https://drive.google.com/uc?id=17J0QcMSYEK9EaeuQWf5HCalkOmcSG4Tb)](https://drive.google.com/file/d/17J0QcMSYEK9EaeuQWf5HCalkOmcSG4Tb/view?usp=sharing) | [![Interpretability Plot 3](https://drive.google.com/uc?id=17Iol63j4LyfqkD2ifz5e00uCSKWQ70zo)](https://drive.google.com/file/d/17Iol63j4LyfqkD2ifz5e00uCSKWQ70zo/view?usp=sharing) | [![Interpretability Plot 4](https://drive.google.com/uc?id=17D9BLRgLr5WZwO2odQeHzoCQiobTaiGi)](https://drive.google.com/file/d/17D9BLRgLr5WZwO2odQeHzoCQiobTaiGi/view?usp=sharing) | [![Interpretability Plot 5](https://drive.google.com/uc?id=17HB9P0xmBnQO7YiHLth-QnGb8VpTOcb6)](https://drive.google.com/file/d/17HB9P0xmBnQO7YiHLth-QnGb8VpTOcb6/view?usp=sharing) |




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

| Frame 1 (Healthy) | Frame 2 (Unhealthy) | Frame 3 (Unhealthy) | Frame 4 (Unhealthy) | Frame 5 (Healthy) |
|-------------------|----------------------|----------------------|----------------------|-------------------|
| [![Frame 1](https://drive.google.com/uc?id=1lr2RddaY5cSrdlGgzRr4Hi7KzO-DxvDD)](https://drive.google.com/file/d/1lr2RddaY5cSrdlGgzRr4Hi7KzO-DxvDD/view?usp=sharing) | [![Frame 2](https://drive.google.com/uc?id=1smCJjNWxy5t0ScDrnMnW2AtGJU5qAfTX)](https://drive.google.com/file/d/1smCJjNWxy5t0ScDrnMnW2AtGJU5qAfTX/view?usp=sharing) | [![Frame 3](https://drive.google.com/uc?id=1vwlhrIAdEQvJrc_MkhaQFU2_Y4vwtCoG)](https://drive.google.com/file/d/1vwlhrIAdEQvJrc_MkhaQFU2_Y4vwtCoG/view?usp=sharing) | [![Frame 4](https://drive.google.com/uc?id=1SFVGJkcaUMIMDqUIEv82VVgmfaNrglFZ)](https://drive.google.com/file/d/1SFVGJkcaUMIMDqUIEv82VVgmfaNrglFZ/view?usp=sharing) | [![Frame 5](https://drive.google.com/uc?id=1fwpkYrg8ktXPEFvj5X7k01qLpw8S_2Cg)](https://drive.google.com/file/d/1fwpkYrg8ktXPEFvj5X7k01qLpw8S_2Cg/view?usp=sharing) |

### Interpretability Plots

| Interpretability Plot 1 | Interpretability Plot 2 | Interpretability Plot 3 | Interpretability Plot 4 | Interpretability Plot 5 |
|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| [![Interpretability Plot 1](https://drive.google.com/uc?id=1PjP4tNw5ZpE6F9dGWBFFtgym9u3as4AM)](https://drive.google.com/file/d/1PjP4tNw5ZpE6F9dGWBFFtgym9u3as4AM/view?usp=sharing) | [![Interpretability Plot 2](https://drive.google.com/uc?id=1uywJFI47j-YczLhyVIfL458P897VCwur)](https://drive.google.com/file/d/1uywJFI47j-YczLhyVIfL458P897VCwur/view?usp=sharing) | [![Interpretability Plot 3](https://drive.google.com/uc?id=16E0isK0LuaYehRTkfpzZ_OeHxIi-6Jz6)](https://drive.google.com/file/d/16E0isK0LuaYehRTkfpzZ_OeHxIi-6Jz6/view?usp=sharing) | [![Interpretability Plot 4](https://drive.google.com/uc?id=1X6-f5E0PJM5wLv3xSFCUQEp0q1Gr0nDH)](https://drive.google.com/file/d/1X6-f5E0PJM5wLv3xSFCUQEp0q1Gr0nDH/view?usp=sharing) | [![Interpretability Plot 5](https://drive.google.com/uc?id=1rqhNziazXCUrCS3lu0OS5NrLjzRnGbcx)](https://drive.google.com/file/d/1rqhNziazXCUrCS3lu0OS5NrLjzRnGbcx/view?usp=sharing) |



