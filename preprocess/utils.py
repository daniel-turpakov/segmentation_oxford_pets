import os
import random
from typing import Optional, Union
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim


def plot_images(images: Union[list[np.ndarray], np.ndarray],
                titles: Optional[list] = None,
                figsize: tuple[int, int] = (12, 8),
                n_rows: Optional[int] = None,
                n_cols: Optional[int] = None) -> None:
    """
    Отображение изображений с учетом масок

    :param images: изображения
    :param titles: названия
    :param figsize: размер изображений
    :param n_rows: количество строк
    :param n_cols: количество столбцов

    :return: None
    """

    n_images = len(images)

    rows = n_rows if n_rows else math.ceil(math.sqrt(n_images))
    cols = n_cols if n_cols else math.ceil(n_images / rows)

    plt.figure(figsize=figsize)
    for i in range(n_images):
        ax = plt.subplot(rows, cols, i + 1)
        if len(images[i].shape) == 2:
            ax.imshow(images[i], cmap='gray')
        else:
            ax.imshow(images[i])

        if titles and i < len(titles):
            plt.title(titles[i])
        plt.axis('off')

    plt.show()


def calculate_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """
    Вычисляет iou

    :param pred_mask: предсказанное значение
    :param true_mask: истинное значение

    :return: iou
    """

    pred_binary = pred_mask != 0
    true_binary = true_mask != 0

    intersection = np.logical_and(pred_binary, true_binary).sum()
    union = np.logical_or(pred_binary, true_binary).sum()

    if union == 0:
        return 0.0

    return intersection / union


def get_scheduler(config, optimizer, loader=None, step=None, gamma=None):
    """
    Устанавливает планировщик для обучения

    :param config: класс с конфигурацией
    :param optimizer: оптимизатор
    :param loader: даталоадер
    :param step: шаг
    :param gamma: гамма

    :return: планировщик для обучения
    """

    if config.scheduler_type == 'exp':
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    elif config.scheduler_type == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
    elif config.scheduler_type == 'cosine':
        T_max = config.n_epochs * len(loader)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        return None


def fix_everything(seed):
    """
    Фиксирует значения для воспроизводимости

    :param seed: семя

    :return: None
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.empty_cache()


def seed_worker():
    """
    Фиксирует семя для воркера

    :return: None
    """

    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
