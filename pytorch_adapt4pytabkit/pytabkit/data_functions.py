import numpy as np

from pytoolbox4dev.base import BaseLogger
logger = BaseLogger(__name__)


def split_concat_dataset_indices(concat_dict):
    """
    Split concatenated dataset indices into train, validation, and test index arrays.

    Given a dict of (split_type, dataset) pairs where each dataset contains features 'X',
    this function calculates index ranges based on dataset lengths and assigns them
    to train, validation, or test splits based on the split_type string content.

    Parameters
    ----------
    concat_list : dict
        Dictionary mapping split_type (str) to dataset dict with 'X' key.
        - key > split_type : str, indicating the dataset split type, e.g., 'train', 'valid', 'test'.
        - value > dataset : dict-like, expected to have a key 'X' representing the feature data.

    Returns
    -------
    train_idxs : np.ndarray
        Array of indices corresponding to the concatenated training data.

    val_idxs : np.ndarray
        Array of indices corresponding to the concatenated validation data.

    test_idxs_dict : dict
        Dictionary mapping test split names to their corresponding index arrays.
    """
    train_idxs, val_idxs, test_idxs_dict = [], [], {}
    start = 0
    for split_type, dataset in concat_dict:.items():
        n = len(dataset['X'])
        end = start + n
        if 'train' in split_type.lower():
            train_idxs.extend(range(start, end))
        elif 'valid' in split_type.lower():
            val_idxs.extend(range(start, end))
        else:
            test_idxs_dict[split_type] = list(range(start, end))
        start = end
    train_idxs = np.array(train_idxs)
    logger.debug(f'Train idxs shape: {train_idxs.shape} (1st index: {train_idxs[0]})')
    val_idxs = np.array(val_idxs)
    logger.debug(f'Valid idxs shape: {val_idxs.shape} (1st index: {val_idxs[0]})')
    for test_type in test_idxs_dict.keys():
        test_idxs_dict[test_type] = np.array(test_idxs_dict[test_type])
        logger.debug(f'Test ({test_type}) idxs shape: {test_idxs_dict[test_type].shape} (1st index: {test_idxs_dict[test_type][0]})')

    result = dict(
        train_idxs=train_idxs,
        val_idxs=val_idxs,
        test_idxs_dict=test_idxs_dict
    )
    return result

def concatenate_datasets(concat_dict):
    """
    Concatenate feature arrays 'X' and label arrays 'y' from multiple datasets.

    Given a dict of (split_type, dataset) pairs, where each dataset is a dictionary
    containing 'X' and 'y' keys, this function concatenates all 'X' arrays and all 'y' arrays
    along the first axis.

    Parameters
    ----------
    concat_list : dict
        Dictionary mapping split_type (str) to dataset dict with 'X' key.
        - key > split_type : str, indicating the dataset split type, e.g., 'train', 'valid', 'test'.
        - value > dataset : dict-like, expected to have a key 'X' representing the feature data.

    Returns
    -------
    X_all : numpy.ndarray
        Concatenated feature array from all datasets.

    y_all : numpy.ndarray
        Concatenated label array from all datasets.
    """
    X_all = np.concatenate([d['X'] for _, d in concat_dict.items()], axis=0)
    y_all = np.concatenate([d['y'] for _, d in concat_dict.items()], axis=0)
    idxs_dict = split_concat_dataset_indices(concat_dict)
    return X_all, y_all, idxs_dict