from typing import Iterable

import numpy as np
import torch


def initial_cross_compatibility(thing_labels: Iterable[int], stuff_labels: Iterable[int]) -> torch.Tensor:
    """
    Initial cross compatibility between instances and semantics

    Args:
        num_labels:     The total number of (semantic) labels in the dataset
        stuff_labels:   List of stuff class labels

    Returns:
        A matrix of size (num_labels + 1, num_labels). Rows represent instances and columns represent semantic labels.
        The last row represents the `no_instance` instance class.
    """
    mat = np.zeros((len(thing_labels) + 1, len(stuff_labels) + 1), dtype=np.float32)

    mat[:, 0] = 1
    mat[-1, :] = 1
    mat[-1, 0] = 0
    # for label in range(num_labels):
    #     if label in stuff_labels:
    #         mat[-1, label] = 1.0  # Attraction between stuff labels and no_instance
    #     else:
    #         mat[label, label] = 1.0  # Attraction between thing labels and the instances of the same class

    return torch.FloatTensor(mat)


def get_compatibility(cross_compatibility_matrix: torch.Tensor, instance_cls_labels: torch.Tensor) -> torch.Tensor:
    """
    Return the compatibility transform between the instances and the semantics given the parameter matrix (returned by
    the initial_cross_compatibility() function).

    Args:
        cross_compatibility_matrix: The (num_labels + 1, num_labels) parameter matrix (obtained via the
                                    initial_cross_compatibility() function)
        instance_cls_labels:            Labels of the instances. Label equal to the value `num_labels` represents the
                                    `no_instance` instance.

    Returns:
        Compatibility matrix of shape len(instance_labels) x num_labels
    """
    return torch.index_select(cross_compatibility_matrix, dim=0, index=instance_cls_labels)
