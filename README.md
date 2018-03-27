# Digit_Classifier

The file MNIST.py trains an optimized Random Forest Multiclass model to classify digits from a data augmented MNIST dataset. The dataset is downloaded through Scikit-Learn at the beginning of the script and cached for later usage.

Data augmentation is performed using a custom transformer that shifts each image one pixel to each direction, essentially extending the training set 400%.

Final model performance measures:

Accuracy=0.9284
F1=0.927395482799
Average Precision=0.928758763403
Average Recall=0.92799758435


## Requirements

Anaconda, which includes all dependencies (numpy, scypy, sklearn)

## Conclusions

Not precisely hot stuff but performant enough for a first try, no?