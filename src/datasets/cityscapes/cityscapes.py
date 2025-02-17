#!/usr/bin/env python

# Copyright (C) 2024. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree

from cityscapesscripts.helpers.labels import labels


class CityscapesBase:
    SPLITS = ["train", "valid", "test", "val"]

    # number of classes without void/unlabeled and license plate (class 34)
    N_CLASSES = [19, 33]

    # 1+33 classes (0: unlabeled)
    CLASS_NAMES_FULL = [label.name for label in labels]
    CLASS_COLORS_FULL = [label.color for label in labels]

    # 1+19 classes (0: void)
    CLASS_NAMES_REDUCED = ["void"] + [
        label.name for label in labels if not label.ignoreInEval
    ]
    CLASS_COLORS_REDUCED = [(0, 0, 0)] + [
        label.color for label in labels if not label.ignoreInEval
    ]
    # forward mapping (0: unlabeled) + 33 classes -> (0: void) + 19 classes
    CLASS_MAPPING_REDUCED = {
        c: labels[c].trainId + 1 if not labels[c].ignoreInEval else 0
        for c in range(1 + 33)
    }

    RGB_DIR = "rgb"

    LABELS_FULL_DIR = "labels_33"
    LABELS_FULL_COLORED_DIR = "labels_33_colored"

    LABELS_REDUCED_DIR = "labels_19"
    LABELS_REDUCED_COLORED_DIR = "labels_19_colored"
