from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.models import Sequential

DATASET_PATH = Path(r"D:\leapGestRecog")

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 25
RANDOM_STATE = 42

VALID_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
}

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def find_subject_folders(dataset_path):
    """
    Find numeric subject folders such as 00, 01, ..., 09.
    """

    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Dataset folder was not found:\n{dataset_path}"
        )

    def get_numeric_folders(path):
        return sorted(
            [
                item
                for item in path.iterdir()
                if item.is_dir() and item.name.isdigit()
            ],
            key=lambda item: item.name,
        )

    subject_folders = get_numeric_folders(dataset_path)

    if subject_folders:
        return dataset_path, subject_folders

    nested_path = dataset_path / "leapGestRecog"

    if nested_path.is_dir():
        subject_folders = get_numeric_folders(nested_path)

        if subject_folders:
            return nested_path, subject_folders

    raise ValueError(
        "No numeric subject folders were found.\n"
        "Expected folders such as 00 through 09 inside:\n"
        f"{dataset_path}"
    )


def find_gesture_names(subject_folders):
    """
    Read gesture names and verify that every subject has the same
    gesture folders.
    """

    first_subject = subject_folders[0]

    gesture_names = sorted(
        [
            item.name
            for item in first_subject.iterdir()
            if item.is_dir()
        ]
    )

    if not gesture_names:
        raise ValueError(
            f"No gesture folders were found inside:\n{first_subject}"
        )

    expected_gestures = set(gesture_names)

    for subject_path in subject_folders[1:]:
        subject_gestures = {
            item.name
            for item in subject_path.iterdir()
            if item.is_dir()
        }

        missing = expected_gestures - subject_gestures
        extra = subject_gestures - expected_gestures

        if missing or extra:
            raise ValueError(
                f"Inconsistent gesture folders in subject "
                f"{subject_path.name}.\n"
                f"Missing: {sorted(missing)}\n"
                f"Extra: {sorted(extra)}"
            )

    return gesture_names

def load_dataset(subject_folders, gesture_names):
    gesture_to_label = {
        gesture_name: label
        for label, gesture_name in enumerate(gesture_names)
    }

    images = []
    labels = []
    subject_groups = []

    skipped_images = 0

    for subject_path in subject_folders:
        print(f"Loading subject: {subject_path.name}")

        for gesture_name in gesture_names:
            gesture_path = subject_path / gesture_name
            label = gesture_to_label[gesture_name]

            for image_path in sorted(gesture_path.iterdir()):
                # Prevent folders from being passed to cv2.imread().
                if not image_path.is_file():
                    continue

                if image_path.suffix.lower() not in VALID_EXTENSIONS:
                    continue

                image = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_GRAYSCALE,
                )

                if image is None:
                    skipped_images += 1
                    print(f"Could not read: {image_path}")
                    continue

                image = cv2.resize(
                    image,
                    (IMG_SIZE, IMG_SIZE),
                )

                image = image.astype(np.float32) / 255.0

                images.append(image)
                labels.append(label)
                subject_groups.append(subject_path.name)

    if not images:
        raise ValueError(
            "No images were loaded. Check the dataset path "
            "and folder structure."
        )

    images = np.asarray(
        images,
        dtype=np.float32,
    )

    # Add grayscale channel dimension.
    images = images[..., np.newaxis]

    labels = np.asarray(
        labels,
        dtype=np.int32,
    )

    subject_groups = np.asarray(subject_groups)

    print("\nDataset Information")
    print("===================")
    print("Total images:", len(images))
    print("Skipped images:", skipped_images)
    print("Image shape:", images.shape)
    print("Label shape:", labels.shape)
    print("Number of classes:", len(gesture_names))
    print("Gesture classes:", gesture_names)

    return images, labels, subject_groups

def split_dataset_by_subject(images, labels, subject_groups):
    """
    Keep different subjects in the training, validation, and testing
    datasets for a more realistic evaluation.
    """

    test_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_validation_indices, test_indices = next(
        test_splitter.split(
            images,
            labels,
            groups=subject_groups,
        )
    )

    validation_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.125,
        random_state=RANDOM_STATE,
    )

    relative_train_indices, relative_validation_indices = next(
        validation_splitter.split(
            images[train_validation_indices],
            labels[train_validation_indices],
            groups=subject_groups[train_validation_indices],
        )
    )

    train_indices = train_validation_indices[
        relative_train_indices
    ]

    validation_indices = train_validation_indices[
        relative_validation_indices
    ]

    print("\nDataset Split")
    print("=============")

    print(
        "Training subjects:",
        sorted(set(subject_groups[train_indices])),
    )

    print(
        "Validation subjects:",
        sorted(set(subject_groups[validation_indices])),
    )

    print(
        "Testing subjects:",
        sorted(set(subject_groups[test_indices])),
    )

    print("Training samples:", len(train_indices))
    print("Validation samples:", len(validation_indices))
    print("Testing samples:", len(test_indices))

    return (
        images[train_indices],
        images[validation_indices],
        images[test_indices],
        labels[train_indices],
        labels[validation_indices],
        labels[test_indices],
    )

def build_model(number_of_classes):
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu",
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu",
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(
            128,
            kernel_size=(3, 3),
            activation="relu",
        ),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),

        Dense(
            128,
            activation="relu",
        ),
        Dropout(0.4),

        Dense(
            number_of_classes,
            activation="softmax",
        ),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

def main():
    dataset_root, subject_folders = find_subject_folders(
        DATASET_PATH
    )

    gesture_names = find_gesture_names(
        subject_folders
    )

    number_of_classes = len(gesture_names)

    print("Dataset root:", dataset_root)

    print(
        "Subject folders:",
        [folder.name for folder in subject_folders],
    )

    print("Gesture classes:", gesture_names)
    print("Number of classes:", number_of_classes)

    images, labels, subject_groups = load_dataset(
        subject_folders,
        gesture_names,
    )

    (
        x_train,
        x_validation,
        x_test,
        y_train,
        y_validation,
        y_test,
    ) = split_dataset_by_subject(
        images,
        labels,
        subject_groups,
    )

    model = build_model(number_of_classes)

    print("\nCNN Model")
    print("=========")
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    model_checkpoint = ModelCheckpoint(
        filepath="best_hand_gesture_model.keras",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1,
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(
            x_validation,
            y_validation,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            early_stopping,
            model_checkpoint,
        ],
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=1,
    )

    prediction_probabilities = model.predict(
        x_test,
        verbose=1,
    )

    predicted_classes = np.argmax(
        prediction_probabilities,
        axis=1,
    )

    print("\nModel Evaluation")
    print("================")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test loss: {test_loss:.6f}")

    class_labels = list(
        range(number_of_classes)
    )

    report = classification_report(
        y_test,
        predicted_classes,
        labels=class_labels,
        target_names=gesture_names,
        digits=4,
        zero_division=0,
    )

    print("\nClassification Report")
    print("=====================")
    print(report)

    matrix = confusion_matrix(
        y_test,
        predicted_classes,
        labels=class_labels,
    )

    print("\nConfusion Matrix")
    print("================")
    print(matrix)

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=gesture_names,
        yticklabels=gesture_names,
    )

    plt.title(
        "Confusion Matrix - Hand Gesture Recognition"
    )
    plt.xlabel("Predicted Gesture")
    plt.ylabel("Actual Gesture")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 6))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 6))

    plt.plot(
        history.history["loss"],
        label="Training Loss",
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss",
    )

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    sample_count = min(
        12,
        len(x_test),
    )

    sample_indices = np.random.choice(
        len(x_test),
        size=sample_count,
        replace=False,
    )

    plt.figure(figsize=(12, 9))

    print("\nSample Predictions")
    print("==================")

    for plot_number, sample_index in enumerate(
        sample_indices,
        start=1,
    ):
        actual_name = gesture_names[
            y_test[sample_index]
        ]

        predicted_name = gesture_names[
            predicted_classes[sample_index]
        ]

        confidence = np.max(
            prediction_probabilities[sample_index]
        ) * 100

        print(
            f"Sample {plot_number}: "
            f"Actual={actual_name}, "
            f"Predicted={predicted_name}, "
            f"Confidence={confidence:.1f}%"
        )

        plt.subplot(
            3,
            4,
            plot_number,
        )

        plt.imshow(
            x_test[sample_index].reshape(
                IMG_SIZE,
                IMG_SIZE,
            ),
            cmap="gray",
        )

        title_color = (
            "green"
            if actual_name == predicted_name
            else "red"
        )

        plt.title(
            f"Actual: {actual_name}\n"
            f"Predicted: {predicted_name}\n"
            f"Confidence: {confidence:.1f}%",
            color=title_color,
            fontsize=9,
        )

        plt.axis("off")

    plt.suptitle(
        "Sample Hand Gesture Predictions",
        fontsize=15,
    )

    plt.tight_layout()
    plt.show()

    model.save(
        "hand_gesture_recognition_model.keras"
    )

    print("\nTask completed successfully.")
    print(
        "Model saved as: "
        "hand_gesture_recognition_model.keras"
    )
    print(
        "Best model saved as: "
        "best_hand_gesture_model.keras"
    )


if __name__ == "__main__":
    main()
