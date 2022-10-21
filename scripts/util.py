from enum import IntEnum
from typing import Literal
import numpy as np
import json

DateFormat = Literal['monthly', 'yearly', 'weekly', 'daily']
EPS = 1e-12


class SuperTopic(IntEnum):
    Interesting = 0
    NotRelevant = 1
    COVID = 2
    Politics = 3
    Movements = 4
    Impacts = 5
    Causes = 6
    Solutions = 7
    Contrarian = 8
    Other = 9


def read_temp_dist(filename) -> tuple[list[str], list[int], np.ndarray]:
    with open(filename) as f:
        data = json.load(f)

    return data['groups'], data.get('topics', []), np.array(data['distribution'])


def read_supertopics(filename):
    with open(filename) as annotations_file:
        annotations = [[0] * 10]  # pre-fill topic 0, as it's not included in annotations
        for line in annotations_file:
            splits = line.split('\t')
            annotation = [0] * 10
            for i, s in enumerate(splits):
                annotation[i] = 1 if s.strip() == 'x' else 0
            annotations.append(annotation)
        return np.array(annotations)


def get_spottopics(distributions: np.ndarray, threshold: float, min_size: int):
    th = (distributions.max(axis=0) / distributions.sum(axis=0)) > threshold
    ms = distributions.sum(axis=0) > min_size
    return np.argwhere(np.all([th, ms], axis=0))


def smooth(array, kernel_size, with_pad=True):
    kernel = np.ones(kernel_size) / kernel_size

    if with_pad:
        padded = [np.pad(row, kernel_size // 2, mode='edge') for row in array]
        smoothed = [np.convolve(row, kernel, mode='same') for row in padded]
        return np.array(smoothed).T[kernel_size // 2:-kernel_size // 2].T

    return np.array([np.convolve(row, kernel, mode='valid') for row in array])
