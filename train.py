"""
train.py
Обучение сверточной нейронной сети (CNN) на датасете MNIST
для распознавания рукописных цифр от 0 до 9.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def load_and_preprocess_data():
    """Загрузка и нормализация данных MNIST."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Нормализация пикселей в диапазон [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Добавление канала (batch, 28, 28, 1) для CNN
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    print(f"Train shape: {x_train.shape}, {y_train.shape}")
    print(f"Test  shape: {x_test.shape}, {y_test.shape}")
    return x_train, y_train, x_test, y_test


def build_model(input_shape=(28, 28, 1), num_classes=10):
    """Построение архитектуры CNN."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Первый блок
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Второй блок
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),

        # Классификатор
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    model.summary()
    return model


def plot_history(history):
    """Графики обучения."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["accuracy"], label="train acc")
    ax1.plot(history.history["val_accuracy"], label="val acc")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history.history["loss"], label="train loss")
    ax2.plot(history.history["val_loss"], label="val loss")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=120)
    plt.show()


def main():
    epochs = 10
    batch_size = 128
    model_path = "mnist_cnn.keras"

    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    model = build_model()

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
    )

    # Оценка на тестовой выборке
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss:     {test_loss:.4f}")

    # Сохранение модели
    model.save(model_path)
    print(f"Model saved to: {os.path.abspath(model_path)}")

    plot_history(history)


if __name__ == "__main__":
    main()