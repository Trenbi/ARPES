"""Обробка ARPES-спектрів нейромережами U-Net, LinkNet і DeepLabV3.

Запуск: python main.py
"""
import os
import sys

# Важливо: onnxruntime імпортується ПЕРШИМ, до PyQt5 і matplotlib.
# У Windows після завантаження бібліотек Qt його DLL може спричиняти аварійне завершення
# (Windows fatal exception: access violation) через конфлікт версій системних DLL.
import onnxruntime  # noqa: F401

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
                             QRadioButton, QSlider, QVBoxLayout, QWidget)

import model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def fix_qt_plugin_path():
    """У Windows Qt може не знайти плагін «windows» (помилка «Could not find the Qt platform plugin»),
    якщо шлях до .venv містить кирилицю. Тому вказуємо папку з плагінами явно."""
    import PyQt5
    base = os.path.dirname(PyQt5.__file__)
    for sub in ("Qt5", "Qt"):
        platforms = os.path.join(base, sub, "plugins", "platforms")
        if os.path.isdir(platforms):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platforms
            break


# Фільтр діалогу відкриття файлу
FILE_FILTER = ("Спектри (*.csv *.txt *.dat *.npy *.npz *.h5 *.hdf5 *.mat *.png *.jpg *.jpeg *.tif *.tiff *.bmp);;"
               "Усі файли (*)")


def read_table(path):
    """CSV / TXT / DAT: роздільник (кома, крапка з комою, табуляція чи пробіли) визначається автоматично."""
    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        sample = file.read(4096)
    counts = {delimiter: sample.count(delimiter) for delimiter in (",", ";", "\t")}
    best = max(counts, key=counts.get)
    data = np.genfromtxt(path, delimiter=best if counts[best] else None)
    if data.ndim == 2 and data.shape[0] > 1 and np.all(np.isnan(data[0])):     # заголовок
        data = data[1:]
    if data.ndim == 2 and data.shape[1] > 1 and np.all(np.isnan(data[:, 0])):  # текстовий перший стовпець
        data = data[:, 1:]
    return data


def read_hdf5(path):
    """HDF5: береться перший числовий набір даних з двома і більше вимірами."""
    try:
        import h5py
    except ImportError:
        raise ValueError("Для файлів HDF5 встановіть пакет h5py: pip install h5py")
    found = []
    with h5py.File(path, "r") as file:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and obj.ndim >= 2 and np.issubdtype(obj.dtype, np.number):
                found.append(name)
        file.visititems(visit)
        if not found:
            raise ValueError("У файлі HDF5 немає числових масивів з двома і більше вимірами")
        return np.asarray(file[found[0]])


def read_mat(path):
    """MATLAB .mat: береться перша числова змінна-матриця (файли версії 7.3 — це HDF5)."""
    from scipy.io import loadmat
    try:
        content = loadmat(path)
    except NotImplementedError:
        return read_hdf5(path)
    for name, value in content.items():
        if (not name.startswith("__") and isinstance(value, np.ndarray) and value.ndim >= 2
                and value.size >= 64 and np.issubdtype(value.dtype, np.number)):
            return value
    raise ValueError("У файлі .mat немає числової матриці")


def read_image(path):
    """PNG / JPG / TIFF / BMP: кольорове зображення переводиться у відтінки сірого."""
    from PIL import Image
    with Image.open(path) as image:
        if image.mode not in ("L", "I", "I;16", "F"):
            image = image.convert("L")
        return np.asarray(image, dtype=np.float64)


def load_spectrum(path):
    """Читає спектр з файлу будь-якого підтримуваного формату.

    Рядки — енергія, стовпці — імпульс. Повертає (двовимірний масив, примітка для підпису).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt", ".dat"):
        data = read_table(path)
    elif ext == ".npy":
        data = np.load(path, allow_pickle=False)
    elif ext == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            data = archive[archive.files[0]]
    elif ext in (".h5", ".hdf5"):
        data = read_hdf5(path)
    elif ext == ".mat":
        data = read_mat(path)
    elif ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        data = read_image(path)
    else:
        raise ValueError(f"Непідтримуваний формат: {ext or 'файл без розширення'}")

    data = np.squeeze(np.asarray(data, dtype=np.float64))
    note = ""
    if data.ndim > 2:                   # тривимірний набір: показуємо перший зріз
        note = f" (масив {data.shape}: показано перший зріз)"
        while data.ndim > 2:
            data = data[0]
    if data.ndim != 2 or min(data.shape) < 8:
        raise ValueError("Спектр має бути двовимірним масивом розміром щонайменше 8×8")
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("У файлі немає числових значень")
    return np.nan_to_num(data, nan=finite.min(), posinf=finite.max(), neginf=finite.min()), note


def draw(figure, spectrum, curve=None):
    """Ліворуч — спектр, праворуч — окремий графік знайденої кривої E(k)."""
    rows, cols = spectrum.shape
    figure.clear()
    ax1 = figure.add_subplot(121)
    im = ax1.imshow(spectrum, cmap="viridis", aspect="auto")
    figure.colorbar(im, ax=ax1)
    ax1.set_title("Спектр")
    ax1.set_xlabel("k, піксель")
    ax1.set_ylabel("E, піксель")

    ax2 = figure.add_subplot(122)
    ax2.set_title("Крива дисперсії")
    ax2.set_xlabel("k, піксель")
    ax2.set_xlim(0, cols - 1)
    ax2.set_ylim(rows - 1, 0)          # ті самі межі й напрямок осей, що й у спектра
    ax2.grid(alpha=0.3)
    if curve is not None:
        k, e = curve
        if len(k):
            ax2.plot(k, e, color="tab:red", linewidth=2)
        else:
            ax2.text(0.5, 0.5, "Дисперсію не знайдено.\nЗменшіть поріг.", ha="center", va="center",
                     transform=ax2.transAxes)
    figure.tight_layout()


def save_results(base_path, curve, figure):
    """Зберігає криву в base_path.csv і обидва графіки в base_path.png."""
    k, e = curve
    np.savetxt(base_path + ".csv", np.column_stack([k, e]), delimiter=",", header="k,E", comments="", fmt="%.4f")
    figure.savefig(base_path + ".png", dpi=150)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Обробка ARPES спектрів")
        self.resize(950, 620)
        self.spectrum = None      # завантажений спектр
        self.prob = None          # карта ймовірностей від моделі
        self.curve = None         # знайдена крива (k, E)
        self.model_paths = {}     # шляхи до файлів моделей

        layout = QVBoxLayout()

        # Вибір файлу
        top = QHBoxLayout()
        open_button = QPushButton("Відкрити файл…")
        open_button.clicked.connect(self.open_file)
        self.file_label = QLabel("Файл не обрано")
        top.addWidget(open_button)
        top.addWidget(self.file_label, 1)
        layout.addLayout(top)

        # Графіки
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)

        # Вибір моделі
        models_row = QHBoxLayout()
        models_row.addWidget(QLabel("Модель:"))
        self.radios = {}
        for name in model.MODEL_FILES:
            radio = QRadioButton(name)
            self.radios[name] = radio
            models_row.addWidget(radio)
        self.radios["U-Net"].setChecked(True)
        models_row.addStretch(1)
        layout.addLayout(models_row)

        # Поріг
        threshold_row = QHBoxLayout()
        self.threshold_label = QLabel("Поріг: 0.70")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(70)
        self.slider.valueChanged.connect(self.threshold_changed)
        threshold_row.addWidget(self.threshold_label)
        threshold_row.addWidget(self.slider, 1)
        layout.addLayout(threshold_row)

        # Кнопки
        buttons = QHBoxLayout()
        self.predict_button = QPushButton("Передбачити")
        self.predict_button.setEnabled(False)
        self.predict_button.clicked.connect(self.predict)
        self.save_button = QPushButton("Зберегти")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self.save)
        buttons.addStretch(1)
        buttons.addWidget(self.predict_button)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)

        self.setLayout(layout)

    def threshold(self):
        return self.slider.value() / 100

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Оберіть спектр", "", FILE_FILTER)
        if not path:
            return
        try:
            spectrum, note = load_spectrum(path)
        except Exception as error:
            QMessageBox.critical(self, "Помилка", f"Не вдалося прочитати файл:\n{error}")
            return
        self.spectrum = spectrum
        self.prob = self.curve = None
        self.file_label.setText(os.path.basename(path) + note)
        self.predict_button.setEnabled(True)
        self.save_button.setEnabled(False)
        draw(self.figure, self.spectrum)
        self.canvas.draw()

    def get_model_path(self, name):
        """Повертає шлях до файлу моделі; якщо його немає в папці models/, пропонує вказати вручну."""
        if name in self.model_paths:
            return self.model_paths[name]
        path = os.path.join(BASE_DIR, "models", model.MODEL_FILES[name])
        if not os.path.isfile(path):
            path, _ = QFileDialog.getOpenFileName(self, f"Файл моделі {name}", "", "ONNX (*.onnx)")
            if not path:
                return None
        self.model_paths[name] = path
        return path

    def predict(self):
        name = next(n for n, radio in self.radios.items() if radio.isChecked())
        path = self.get_model_path(name)
        if path is None:
            return
        # Розрахунок може тривати до хвилини: показуємо це на кнопці й курсором
        self.predict_button.setEnabled(False)
        self.predict_button.setText("Триває розрахунок…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            self.prob = model.predict(name, path, self.spectrum)
        except Exception as error:
            self.prob = None
            QMessageBox.critical(self, "Помилка", f"Не вдалося запустити модель {name}:\n{error}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.predict_button.setText("Передбачити")
            self.predict_button.setEnabled(True)
        self.update_curve()

    def threshold_changed(self, value):
        self.threshold_label.setText(f"Поріг: {value / 100:.2f}")
        if self.prob is not None:      # мережу повторно не запускаємо, лише перераховуємо криву
            self.update_curve()

    def update_curve(self):
        self.curve = model.find_curve(self.prob, self.threshold())
        draw(self.figure, self.spectrum, self.curve)
        self.canvas.draw()
        self.save_button.setEnabled(len(self.curve[0]) > 0)

    def save(self):
        path, _ = QFileDialog.getSaveFileName(self, "Зберегти результат", "result.csv", "CSV (*.csv)")
        if not path:
            return
        base = os.path.splitext(path)[0]
        save_results(base, self.curve, self.figure)
        QMessageBox.information(self, "Збережено", f"Збережено файли:\n{base}.csv\n{base}.png")


if __name__ == "__main__":
    fix_qt_plugin_path()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
