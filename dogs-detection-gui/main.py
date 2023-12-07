import base64
import io
from functools import partial
from pathlib import Path
from typing import List

import PySimpleGUI as sg
import cv2
import numpy as np
from PIL import Image

from constants import CLASSES_NAMES_MAPPING, IMAGE_SIZE
from dogs_detection_onnx import get_detections, load_default_model, vis


class DetectionRecord:
    """Класс которых харнит изображение и результат детекции для него."""

    def __init__(self, record_idx, image, bboxes, scores, cls_ids):
        self.image = image.copy()
        self.record_idx = record_idx
        self.bboxes = bboxes
        self.scores = scores
        self.cls_ids = cls_ids
        assert len(bboxes) == len(scores)
        assert len(bboxes) == len(cls_ids)

        self.ids = np.arange(len(bboxes)) + 1
        self.size = len(bboxes)
        self.do_filtering = False
        self.filtering_cls_id = 0
        draw_number(self.image, self.record_idx)

    def vis_detections_by_threshold(self, threshold):
        """Изобразить все детекции."""
        bboxes, scores, cls_ids, ids = self.filter_by_class()
        mask = scores > threshold
        bboxes = bboxes[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]
        ids = ids[mask]
        vis_img = vis(self.image, bboxes, scores, cls_ids, ids)
        return vis_img

    def vis_single_detection(self, idx):
        """Изобразить одну детекцию."""
        if idx < 1 or idx > self.size:
            raise KeyError(f"Wrong index: {idx}")
        bboxes, scores, cls_ids, ids = self.filter_by_class()
        mask = ids == idx
        assert mask.sum() <= 1
        bboxes = bboxes[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]
        ids = ids[mask]
        vis_img = vis(self.image, bboxes, scores, cls_ids, ids)
        return vis_img

    def get_statistics(self, threshold):
        """Получить статистику по изображению."""
        mask_human = self.cls_ids == 0
        mask_dog = self.cls_ids == 16
        confident_count_human = np.sum(self.scores[mask_human] > threshold)
        confident_count_dog = np.sum(self.scores[mask_dog] > threshold)
        confident_count = np.sum(self.scores > threshold)
        unconfident_count = self.size - confident_count
        return confident_count_human, confident_count_dog, unconfident_count

    def filter_by_class(self):
        """Отфильтровать по классам."""
        if not self.do_filtering:
            return self.bboxes, self.scores, self.cls_ids, self.ids
        cls_id = self.filtering_cls_id
        if cls_id is None:
            return np.array([]), np.array([]), np.array([]), np.array([])
        assert isinstance(cls_id, int)
        mask = self.cls_ids == cls_id
        return (
            self.bboxes[mask],
            self.scores[mask],
            self.cls_ids[mask],
            self.ids[mask],
        )


def convert_to_bytes(file_or_bytes, resize=None):
    """
    Prepare image before draw on PysimpleGUI element.

    Convert into bytes and optionally resize an image that
    is a file or a base64 bytes object.
    Turns into  PNG format in the process so that can be displayed by tkinter.

    file_or_bytes: either a string filename or a bytes base64 image object
    resize:  optional new size

    return: (bytes) a byte-string object
    """
    if isinstance(file_or_bytes, str):
        img = Image.open(file_or_bytes)
    elif isinstance(file_or_bytes, np.ndarray):
        img = Image.fromarray(file_or_bytes)
    else:
        try:
            img = Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = Image.open(dataBytesIO)
            print(e)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height / cur_height, new_width / cur_width)
        img = img.resize(
            (int(cur_width * scale), int(cur_height * scale)),
            Image.ANTIALIAS
        )
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()


def draw_number(img, idx):
    """Нарисовать номер на изображении."""
    shift_horizontal = 100
    shift_vertical = 30
    if idx is not None:
        assert isinstance(idx, int)
        h, w = img.shape[:2]
        cv2.putText(
            img,
            f"# {idx}",
            (w - shift_horizontal, h - shift_vertical),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            thickness=7,
        )
        cv2.putText(
            img,
            f"# {idx}",
            (w - shift_horizontal, h - shift_vertical),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            thickness=3,
        )


def filter_detections_by_target_class(
        bboxes, scores, cls_ids
):
    mask = np.isin(cls_ids, list(CLASSES_NAMES_MAPPING.keys()))
    return bboxes[mask], scores[mask], cls_ids[mask]


def load_and_process_images_from_dir(directory, model):
    """Загрузка файлов из папки и обработка с помощью нейросети"""
    directory = Path(directory)
    records = []
    for i, fpath in enumerate(directory.glob("*[.jpg|.jpeg|.png]"), 1):
        image = cv2.imread(str(fpath))
        if image is None:
            raise RuntimeError(f'Can not open image: {fpath}')
        bboxes, scores, cls_ids = get_detections(model, image)
        bboxes, scores, cls_ids = filter_detections_by_target_class(
            bboxes, scores, cls_ids
        )
        records.append(
            DetectionRecord(
                i,
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                bboxes,
                scores,
                cls_ids
            )
        )
    return records


def filter_records_without_detections(records: List[DetectionRecord]):
    """Получить все записи где есть хоть одна детекция."""
    return [rec for rec in records if rec.size > 0]


def create_layout():
    """Создать элементы и задать их взаимное расположение на окне."""
    # Левая колонка
    left_col = [
        [
            sg.Text("Folder"),
            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
        ],
        [sg.Image(key="-IMAGE-", size=IMAGE_SIZE)],
        [
            sg.Button("<", key="-IMGPREV-", size=(2, 2)),
            sg.Button(
                "Убрать / Вернуть выделение\nопознанных образов",
                key="-DRAW-DETECTIONS-FLAG-",
                size=(25, 2),
            ),
            sg.Text("Поиск по\nномеру\nкартинки"),
            sg.InputText(key="-INPNUMIMG-", size=(10, 1), enable_events=True),
            sg.Text(
                "Поиск по\nномеру\nобраза",
            ),
            sg.InputText(key="-INPNUMDET-", size=(10, 1), enable_events=True),
            sg.Button(">", key="-IMGNEXT-", size=(2, 2)),
        ],
    ]

    frame1 = sg.Frame(
        "Сводка текущего изображения",
        [
            [
                sg.Text(
                    "\nЧисло распознанных людей",
                    size=(35, 3),
                    background_color="orange",
                ),
                sg.Text(
                    size=(15, 3),
                    key="-HUMANS-NUM-",
                    text_color="black",
                    justification="center",
                    background_color="white",
                ),
            ],
            [
                sg.Text(
                    "\nЧисло распознанных собак",
                    size=(35, 3),
                    background_color="orange",
                ),
                sg.Text(
                    size=(15, 3),
                    key="-DOGS-NUM-",
                    text_color="black",
                    justification="center",
                    background_color="white",
                ),
            ],
            [
                sg.Text(
                    "Количество объектов с низкой достоверностью",
                    size=(35, 3),
                    background_color="red",
                ),
                sg.Text(
                    size=(15, 3),
                    key="-MISSED-NUM-",
                    text_color="black",
                    justification="center",
                    background_color="white",
                ),
            ],
        ],
    )

    frame2 = sg.Frame(
        "Настройка отображения изображений",
        [
            [
                sg.Text(
                    "Изображения, содержащие только\nраспознанные образы",
                    size=(35, 3),
                    background_color="green",
                ),
                sg.Checkbox(
                    "",
                    size=(5, 3),
                    key="-DETECTED-ONLY-",
                    default=False,
                    text_color="black",
                    background_color="white",
                    enable_events=True,
                ),
            ],
            [
                sg.Text("Всего изображений: ", size=(35, 1), background_color="green"),
                sg.Text(
                    size=(5, 1),
                    key="-TOTAL-IMGS-",
                    text_color="black",
                    justification="center",
                    background_color="white",
                ),
            ],
            [
                sg.Text("Отображать", size=(15, 1), background_color="LightGreen"),
                sg.OptionMenu(
                    ["Всех", "Собак", "Людей", "Никого"],
                    key="-FILTER-CLASSES-",
                    default_value="Всех",
                ),
            ],
            [
                sg.Text(
                    "Отображать образы с точностью распознавания от",
                    size=(50, 1),
                    background_color="Orange",
                )
            ],
            [
                sg.Text("0%"),
                sg.Slider(
                    range=(0, 100),
                    resolution=10,
                    tick_interval=10,
                    key="-THRESHOLD-",
                    default_value=50,
                    orientation="h",
                    size=(40, 20),
                    enable_events=True,
                ),
                sg.Text("100%"),
            ],
        ],
    )

    right_col = [[frame1], [frame2]]

    layout = [
        [sg.VPush()],
        [
            sg.Push(),
            sg.Column(left_col, element_justification="c"),
            sg.Push(),
            sg.VSeperator(),
            sg.Push(),
            sg.Column(right_col, element_justification="l"),
            sg.Push(),
        ],
        [sg.VPush()],
    ]
    return layout


def load_and_process(window, directory, model):
    """Открыть папку с картинками и обработать нейросетью."""
    records = load_and_process_images_from_dir(directory, model)
    cur_idx = 0
    max_idx = len(records) - 1
    window["-IMAGE-"].update(
        data=convert_to_bytes(
            records[cur_idx].vis_detections_by_threshold(0.5), resize=IMAGE_SIZE
        )
    )
    return records, cur_idx, max_idx


def draw_image_all_bboxes(window, record: DetectionRecord, threshold, draw_boxes=True):
    """Отобразить все детекции."""
    if threshold is None:
        return
    if draw_boxes:
        vis_img = record.vis_detections_by_threshold(threshold)
    else:
        vis_img = record.image.copy()
    window["-IMAGE-"].update(
        data=convert_to_bytes(vis_img, resize=IMAGE_SIZE),
        size=IMAGE_SIZE,
    )
    update_image_statistics(window, record, threshold)
    window.refresh()


def draw_image_single_bbox(
    window, record: DetectionRecord, bbox_idx=None, draw_boxes=True
):
    """Отобразить одну детекцию."""
    if draw_boxes:
        vis_img = record.vis_single_detection(bbox_idx)
    else:
        vis_img = record.image.copy()
    if vis_img is None:
        return
    window["-IMAGE-"].update(
        data=convert_to_bytes(vis_img, resize=IMAGE_SIZE),
        size=IMAGE_SIZE,
    )
    window.refresh()


def clear_text_boxes(window):
    """Очистить поля ввода."""
    window["-INPNUMDET-"].update("")
    window["-INPNUMIMG-"].update("")


def update_image_statistics(window, record: DetectionRecord, threshold):
    """Обновить статистики для текущего изображения."""
    found_humans, found_dogs, missed = record.get_statistics(threshold)
    window["-HUMANS-NUM-"].update(f"\n{found_humans}")
    window["-DOGS-NUM-"].update(f"\n{found_dogs}")
    window["-MISSED-NUM-"].update(f"\n{missed}")


def option_menu_callback(window, var, index, mode):
    """Колбэк для генерации событий выпадающим списком."""
    window.write_event_value(
        "-FILTER-CLASSES-", window["-FILTER-CLASSES-"].TKStringVar.get()
    )


def main():
    """
    Основная функция.

    Создание окна приложения и обработка событий (нажатие кнопок и т.д.)
    """
    model = load_default_model()
    records_all, records, records_with_detections = [], [], []
    max_idx = 0
    cur_idx = 0
    flag_draw_detections = True
    setup_made = False

    window = sg.Window(
        "Детекция объектов", create_layout(), resizable=True, finalize=True,
    )
    # Выход по ESC
    window.bind("<Escape>", "-ESCAPE-")

    while True:
        event, values = window.read()
        if not setup_made:
            window["-FILTER-CLASSES-"].TKStringVar.trace(
                "w", partial(option_menu_callback, window)
            )
            setup_made = True
        print("EVENT", event)
        if event in (sg.WIN_CLOSED, "Exit", '-ESCAPE-'):
            break
        if event == "-FOLDER-":
            records_all, cur_idx, max_idx = load_and_process(
                window, values["-FOLDER-"], model
            )
            records_with_detections = filter_records_without_detections(
                records_all
            )
            records = records_all
            draw_image_all_bboxes(
                window, records[cur_idx], values["-THRESHOLD-"] / 100
            )
            window["-TOTAL-IMGS-"].update(len(records))
            clear_text_boxes(window)
        if len(records) == 0:
            continue
        if event == "-THRESHOLD-":
            draw_image_all_bboxes(
                window,
                records[cur_idx],
                values["-THRESHOLD-"] / 100,
                draw_boxes=flag_draw_detections,
            )
            clear_text_boxes(window)
        if event == "-IMGPREV-":
            if cur_idx == 0:
                continue
            cur_idx -= 1
            draw_image_all_bboxes(
                window,
                records[cur_idx],
                values["-THRESHOLD-"] / 100,
                draw_boxes=flag_draw_detections,
            )
            clear_text_boxes(window)
        if event == "-IMGNEXT-":
            if cur_idx == max_idx:
                continue
            cur_idx += 1
            draw_image_all_bboxes(
                window,
                records[cur_idx],
                values["-THRESHOLD-"] / 100,
                draw_boxes=flag_draw_detections,
            )
            clear_text_boxes(window)
        if event == "-INPNUMDET-":
            if values["-INPNUMDET-"] == "":
                draw_image_all_bboxes(
                    window,
                    records[cur_idx],
                    values["-THRESHOLD-"] / 100,
                    draw_boxes=flag_draw_detections,
                )
            else:
                try:
                    draw_image_single_bbox(
                        window,
                        records[cur_idx],
                        bbox_idx=int(values["-INPNUMDET-"])
                    )
                except ValueError as e:
                    sg.PopupError(
                        f"Error: Неправильное значение в поле. Введите число."
                    )
                    continue
                except KeyError as e:
                    sg.PopupError(f"Error: Образ с таким номером не найден!")
                    continue
        if event == "-INPNUMIMG-":
            if values["-INPNUMIMG-"] == "":
                draw_image_all_bboxes(
                    window,
                    records[cur_idx],
                    values["-THRESHOLD-"] / 100,
                    draw_boxes=flag_draw_detections,
                )
            else:
                try:
                    new_idx = int(values["-INPNUMIMG-"]) - 1
                    if new_idx < 0:
                        raise IndexError
                    draw_image_all_bboxes(
                        window,
                        records[new_idx],
                        values["-THRESHOLD-"] / 100,
                        draw_boxes=flag_draw_detections,
                    )
                    cur_idx = new_idx
                except IndexError as e:
                    sg.PopupError(
                        "Error: Изображения с таким номером не найдено. "
                        f"Введите число от 1 до {max_idx+1}."
                    )
                except ValueError as e:
                    sg.PopupError(
                        f"Error: Неправильное значение в поле. Введите число."
                    )
                except KeyError as e:
                    sg.PopupError(f"Error: Образ с таким номером не найден!")
        if event == "-DETECTED-ONLY-":
            if values["-DETECTED-ONLY-"]:
                records = records_with_detections
                cur_idx = 0
                max_idx = len(records_with_detections) - 1
                draw_image_all_bboxes(
                    window,
                    records[cur_idx],
                    values["-THRESHOLD-"] / 100,
                    draw_boxes=flag_draw_detections,
                )
                clear_text_boxes(window)
            else:
                records = records_all
                cur_idx = 0
                max_idx = len(records_all) - 1
                draw_image_all_bboxes(
                    window,
                    records[cur_idx],
                    values["-THRESHOLD-"] / 100,
                    draw_boxes=flag_draw_detections,
                )
            window["-TOTAL-IMGS-"].update(len(records))
        if event == "-FILTER-CLASSES-":
            if values["-FILTER-CLASSES-"] == "Всех":
                for rec in records:
                    rec.do_filtering = False
            elif values["-FILTER-CLASSES-"] == "Никого":
                for rec in records:
                    rec.do_filtering = True
                    rec.filtering_cls_id = None
            elif values["-FILTER-CLASSES-"] == "Собак":
                for rec in records:
                    rec.do_filtering = True
                    rec.filtering_cls_id = 16
            elif values["-FILTER-CLASSES-"] == "Людей":
                for rec in records:
                    rec.do_filtering = True
                    rec.filtering_cls_id = 0
            draw_image_all_bboxes(
                window,
                records[cur_idx],
                values["-THRESHOLD-"] / 100,
                draw_boxes=flag_draw_detections,
            )
            clear_text_boxes(window)
        if event == "-DRAW-DETECTIONS-FLAG-":
            flag_draw_detections = not flag_draw_detections
            draw_image_all_bboxes(
                window,
                records[cur_idx],
                values["-THRESHOLD-"] / 100,
                draw_boxes=flag_draw_detections,
            )
    window.close()


if __name__ == "__main__":
    main()
