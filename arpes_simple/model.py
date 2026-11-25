"""Робота з ONNX-моделями: передбачення та пошук кривої дисперсії.

Саме передбачення виконує файл onnx_predict_module.py: він нарізає спектр на вікна 128×128,
проганяє їх через мережу і склеює карту ймовірностей. Цей файл лише готує спектр до такої обробки
і будує криву за готовою картою.
"""
import numpy as np

import onnx_predict_module as opm

# Назви моделей на екрані → імена файлів у папці models/
MODEL_FILES = {
    "U-Net": "unet.onnx",
    "LinkNet": "linknet.onnx",
    "DeepLabV3": "deeplabv3.onnx",
}

# Моделі навчені в різних бібліотеках і мають різний формат входу та виходу

PREDICTORS = {
    "U-Net": opm.predict_band,
    "LinkNet": opm.predict_band,
    "DeepLabV3": opm.predict_band_deeplab,
}

WINDOW = opm.nn_size[0]          # розмір вікна, на якому навчалися мережі (128 × 128)
STEP = 8                         # крок вікна в пікселях: менший крок — точніше, але повільніше
BATCH_SIZE = 32                  # скільки вікон мережа обробляє за один раз
MAX_MEMORY_BYTES = 1.5e9         # орієнтовний ліміт пам'яті на передбачення (≈ 1,5 ГБ)


def padded_size(n, step):
    """Розмір сторони, з яким модуль покриває кожен піксель.

    Модуль потребує сторону більшу за 128, а коли (n − 128) ділиться на крок націло,
    не додає останнє вікно, і крайні пікселі лишаються без передбачення.
    Тому в такому разі сторона збільшується на один крок: «пропущеною» стає лише додана смуга.
    """
    n = max(n, WINDOW + 1)
    return n + step if (n - WINDOW) % step == 0 else n


def choose_step(rows, cols):
    """Крок вікна: STEP, а для дуже великих спектрів — більший, щоб вистачило пам'яті
    (якщо не вистачає й тоді, виникає ValueError).

    Пам'ять іде на дві речі: проміжний float64-масив (рядки × стовпці × кількість рядків вікон)
    і самі вікна разом з відповідями мережі (приблизно утричі більше за розмір вхідних вікон).
    """
    step = STEP
    while step < WINDOW:
        pr, pc = padded_size(rows, step), padded_size(cols, step)
        layers_r, layers_c = (pr - WINDOW) // step + 2, (pc - WINDOW) // step + 2
        buffer = pr * pc * layers_r * 8
        windows = layers_r * layers_c * WINDOW * WINDOW * 4 * 3
        if buffer + windows <= MAX_MEMORY_BYTES:
            return step
        step *= 2
    raise ValueError("Спектр завеликий для обробки. Зменште його розмір (наприклад, обріжте область інтересу).")


def predict(name, model_path, spectrum):
    """Повертає карту ймовірностей розміром як у спектра (значення від 0 до 1).

    name — «U-Net», «LinkNet» або «DeepLabV3», model_path — шлях до файлу .onnx.
    """
    rows, cols = spectrum.shape
    span = spectrum.max() - spectrum.min()
    if span == 0:
        raise ValueError("Спектр однорідний: усі значення однакові")

    image = opm.normalize(spectrum.astype(np.float64))      # мін-макс до діапазону [0, 1]
    step = choose_step(rows, cols)
    pad_r, pad_c = padded_size(rows, step) - rows, padded_size(cols, step) - cols
    image = np.pad(image, ((0, pad_r), (0, pad_c)), mode="edge").astype(np.float32)[:, :, None]  # (рядки, стовпці, 1)

    prob = PREDICTORS[name](image, model_path, step=(step, step), batch_size=BATCH_SIZE)
    prob = prob[:rows, :cols]                                # прибираємо додану смугу

    if prob.min() < 0 or prob.max() > 1:                     # якщо модель віддала не ймовірності, а логіти
        prob = 1 / (1 + np.exp(-prob))
    return prob


def find_curve(prob, threshold):
    """Крива дисперсії: у кожному стовпці — середній рядок пікселів з ймовірністю ≥ порогу.

    Повертає (k, E) — номери стовпців і положення по рядках (у пікселях).
    """
    k, e = [], []
    rows = np.arange(prob.shape[0])
    threshold = max(threshold, 1e-6)       # при нульовому порозі ваги могли б бути нульовими
    for col in range(prob.shape[1]):
        mask = prob[:, col] >= threshold
        if mask.any():
            k.append(col)
            e.append(np.average(rows[mask], weights=prob[mask, col]))
    return np.array(k), np.array(e)
