from itertools import islice

import numpy as np


def ins_to_sem_compatibility(sem_labels_for_instances, num_sem_labels, stuff_sem_cls_ids, stuff_penalisation=1.0):
    """
    Returns the compatibility matrix for the instance_labels -> semantic_labels bipartite potentials in BCRF.

    Args:
        sem_labels_for_instances:   Semantic labels of the instances. 0th instance must be 'no_instance' with label -1
        num_sem_labels:             The total number of semantic labels used
        stuff_sem_cls_ids:          List of semantic labels for the 'stuff' classes (no instances for these classes)
        stuff_penalisation:         Relative strength of the association between 'stuff' classes and 'no_instance'
                                    object instance (instance label 0)

    Returns:
        A matrix of shape (num_instances, num_sem_labels), where the entry (i, j) contains the connection strength
        between the i th instance and the j the semantic label.
    """
    mat = np.zeros((len(sem_labels_for_instances), num_sem_labels), dtype=np.float32)
    assert sem_labels_for_instances[0] == -1  # First instance must be 'no_instance'

    # Attraction between an instance and its semantic class
    for inst_lbl, sem_lbl in islice(enumerate(sem_labels_for_instances), 1, None):
        mat[inst_lbl, sem_lbl] = 1.0  # TODO(sadeep) Learn this as a vector of size len(thing_sem_cls_ids)

    # Attraction between `no_instance` and stuff classes
    if stuff_sem_cls_ids is not None:
        for stuff_id in stuff_sem_cls_ids:
            mat[0, stuff_id] = stuff_penalisation  # TODO(sadeep) Learn this as a vector of size len(stuff_sem_cls_ids)

    return mat
