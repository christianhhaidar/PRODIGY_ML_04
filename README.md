# Hand Gesture Recognition Using CNN

A convolutional neural network that classifies ten hand gestures from the
[LeapGestRecog dataset](https://www.kaggle.com/gti-upm/leapgestrecog).

The model uses grayscale `64 x 64` images and evaluates generalization with a
subject-independent split: people used for testing do not appear in training.

## Results

- Test accuracy: **78.03%**
- Macro precision: **82.65%**
- Macro recall: **78.02%**
- Macro F1-score: **78.10%**
- Training images: **14,000**
- Validation images: **2,000**
- Testing images: **4,000**

The best validation result occurred at epoch 3. Early stopping restored those
weights after validation performance began to decline.

## Result previews

### Confusion matrix

![Confusion matrix](results/confusion_matrix.jpg)

### Training and validation accuracy

![Accuracy curve](results/accuracy_curve.jpg)

### Training and validation loss

![Loss curve](results/loss_curve.jpg)

### Sample predictions

![Sample predictions](results/sample_predictions.jpg)

## Repository structure

```text
.
|-- gesture_recognition_task4.py
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- models/
|   `-- README.md
`-- results/
    |-- confusion_matrix.jpg
    |-- accuracy_curve.jpg
    |-- loss_curve.jpg
    |-- sample_predictions.jpg
    `-- terminal_output.txt
```

## Setup

1. Install Python 3.10, 3.11, or 3.12.
2. Download and extract the LeapGestRecog dataset.
3. Change `DATASET_PATH` in `gesture_recognition_task4.py` if necessary:

   ```python
   DATASET_PATH = Path(r"D:\leapGestRecog")
   ```

4. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

5. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

6. Train and evaluate the model:

   ```powershell
   python gesture_recognition_task4.py
   ```

The script displays the classification report and confusion matrix in the
terminal, opens the plots, and creates:

- `best_hand_gesture_model.keras`
- `hand_gesture_recognition_model.keras`

## Dataset

The dataset is not included in this repository because it contains 20,000
images. Download it directly from
[Kaggle](https://www.kaggle.com/gti-upm/leapgestrecog).

## Notes

- TensorFlow 2.11 and newer do not provide native-Windows NVIDIA GPU support.
  The script still works on the CPU.
- The trained model is designed for infrared-style Leap Motion images.
  Performance on ordinary webcam photographs may be lower without retraining.
- The subject-independent split produces a more realistic result than randomly
  mixing images of the same people across training and testing.
