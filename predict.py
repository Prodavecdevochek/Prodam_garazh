"""
predict.py
Загрузка обученной модели и предсказание цифры на изображении.
Поддерживает:
  - случайный пример из MNIST
  - собственное изображение (PNG/JPG) через аргумент --image
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import matplotlib.pyplot as plt


def load_model(path: str = "mnist_cnn.keras"):
    print(f"Loading model from: {path}")
    return keras.models.load_model(path)


def preprocess_image(img_path: str) -> np.ndarray:
    """
    Загружает изображение, приводит к 28x28, инвертирует,
    если фон светлый (как на бумаге), и нормализует.
    """
    img = Image.open(img_path).convert("L")  # grayscale
    img = img.resize((28, 28))
    arr = np.array(img, dtype="float32")

    # Если средняя яркость высокая — инвертируем (MNIST: белый текст на чёрном фоне)
    if arr.mean() > 127:
        arr = 255.0 - arr

    arr = arr / 255.0
    return arr[np.newaxis, ..., np.newaxis]  # (1, 28, 28, 1)


def predict_from_file(model, img_path: str):
    x = preprocess_image(img_path)
    probs = model.predict(x, verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(probs[digit]) * 100
    return digit, confidence, x[0, ..., 0]


def predict_random_sample(model, x_test, y_test):
    idx = np.random.randint(0, len(x_test))
    x = x_test[idx:idx+1]
    y = y_test[idx]
    probs = model.predict(x, verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(probs[digit]) * 100
    return digit, confidence, x[0, ..., 0], y


def main():
    parser = argparse.ArgumentParser(description="Распознавание цифры")
    parser.add_argument("--image", type=str, default=None,
                        help="Путь к изображению цифры (28x28 или любое)")
    parser.add_argument("--model", type=str, default="mnist_cnn.keras",
                        help="Путь к сохранённой модели")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image:
        digit, conf, img = predict_from_file(model, args.image)
        true_label = "?"
    else:
        (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
        digit, conf, img, true_label = predict_random_sample(model, x_test, y_test)

    print(f"\nПредсказанная цифра: {digit}")
    print(f"Уверенность:         {conf:.2f}%")
    if args.image is None:
        print(f"Истинная цифра:      {true_label}")

    plt.imshow(img, cmap="gray")
    plt.title(f"Predicted: {digit}  ({conf:.1f}%)")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()