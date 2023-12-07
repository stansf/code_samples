import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

from constants import CLASSES_NAMES_MAPPING, IMAGE_SIZE


def parse_args():
    """Парсинг аргуменов командной строки"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default='yolox_nano_dogs.onnx',
        help="onnx model",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to the output image.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Score threshould to filter the result.",
    )
    return parser.parse_args()


def preprocess(img, input_size):
    """Предобработка входного изображения"""
    # Создание пустого изображения нужного размера
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3),
                             dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    # Масштабирование и "вклеивание" изображения в пустое.
    # Это сделано, чтобы сохранить соотношение сторон на картинке и дополнить
    # пустыми пикселями по краям до нужного размера
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    # Поместить размерность каналов в начало
    # HxWxC -> CxHxW
    padded_img = padded_img.transpose((2, 0, 1))
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def nms(boxes, scores, nms_thr):
    """
    Non-maximum supression для одного класса.

    Составляет список индексов, в которых находятся подходящие баундинг боксы,
    убирая дублирующие.
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """
    NMS для множества классов.

    Может быть также полезна в будущем, если добавятся новые классы.
    """
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep, None],
             valid_cls_inds[keep, None]], 1
        )
    return dets


def postprocess(outputs, img_size):
    """
    Постобработка выходов сети.

    Масштабирование из нормированных координат [0, 1] в абсолютные
    координаты в размерах изображения.
    """
    grids = []
    expanded_strides = []

    strides = [8, 16, 32]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def vis(img, boxes, scores, cls_ids, idxs, conf=0.5):
    """Визуализация детекций  с вероятностью."""
    img = img.copy()
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        # if score < conf:
        #     continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (255, 0, 0)
        idx = idxs[i]
        name = CLASSES_NAMES_MAPPING[cls_id]
        text = f'{idx} | {name}:{score * 100:.1f}%'
        txt_color = (0, 0, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.8, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.8, txt_color,
                    thickness=1)
    return img


def load_default_model():
    return onnxruntime.InferenceSession('yolox_nano.onnx')


def get_detections(onnx_session, image):
    # Водной размер изображения
    input_shape = (416, 416)
    # Выполнить предарительную обработку перед подачей на вход сети
    img, ratio = preprocess(image, input_shape)
    ort_inputs = {onnx_session.get_inputs()[0].name: img[None, :, :, :]}
    # Инференс сети
    output = onnx_session.run(None, ort_inputs)

    # Выполнить постобработку результатов
    predictions = postprocess(output[0], input_shape)[0]

    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio

    # Ghbvytbnm NMS для того чтобы убрать дубликаты
    dets = multiclass_nms_class_agnostic(boxes_xyxy, scores, nms_thr=0.45,
                                         score_thr=0.1)
    final_boxes, final_scores, final_cls_inds = (np.array([]), np.array([]),
                                                 np.array([]))
    if dets is not None:
        final_boxes, final_scores, final_cls_inds = \
            dets[:, :4], dets[:, 4], dets[:, 5]
    return final_boxes, final_scores, final_cls_inds

#
# def main(args):
#     # Водной размер изображения
#     input_shape = (416, 416)
#     # открыть изображение
#     if args.image is None:
#         image_path = input('Введите путь к картинке: ')
#     else:
#         image_path = args.image
#     print(f'Путь к картинке: {image_path}')
#     image_path = str(Path(image_path))
#     if not os.path.exists(image_path):
#         raise RuntimeError(f'Изображение {image_path} не найдено.')
#     origin_img = cv2.imread(image_path)
#     if origin_img is None:
#         print(f'Что-то пошло не так. Не получилось открыть файл {image_path}')
#     # Выполнить предарительную обработку перед подачей на вход сети
#     img, ratio = preprocess(origin_img, (416,416))
#
#     # Открыть ONNX модель
#     session = onnxruntime.InferenceSession(args.model)
#     if session is None:
#         print(f'Что-то пошло не так. Проблема с файлов весов {args.model}')
#     ort_inputs = {session.get_inputs()[0].name: img[None, :, :, :]}
#
#     # Инференс сети
#     output = session.run(None, ort_inputs)
#
#     # Выполнить постобработку результатов
#     predictions = postprocess(output[0], input_shape)[0]
#
#     boxes = predictions[:, :4]
#     scores = predictions[:, 4:5] * predictions[:, 5:]
#
#     boxes_xyxy = np.ones_like(boxes)
#     boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
#     boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
#     boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
#     boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
#     boxes_xyxy /= ratio
#
#     # Ghbvytbnm NMS для того чтобы убрать дубликаты
#     dets = multiclass_nms_class_agnostic(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
#     if dets is not None:
#         final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
#         origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
#                          conf=args.threshold)
#
#     # Сохранить результат на диск
#     if args.output is None:
#         output_path = input('Введите путь, куда сохранить результат картинке (нажимите Enter, чтобы сохранить в текущую папку с именем results.jpg): ')
#         if output_path == '':
#             output_path = 'results.jpg'
#     else:
#         output_path = args.image
#     cv2.imwrite(output_path, origin_img)
#
#
# if __name__ == '__main__':
#     args = parse_args()
#     main(args)
