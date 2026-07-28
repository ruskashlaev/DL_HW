import random

import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    K = 120000
    waveforms = []
    for elem in dataset_items:
        wf = elem["data_object"]
        if wf.shape[0] >= K:
            n = random.randint(0, wf.shape[0] - K)
            wf = wf[n : n + K]
        else:
            repeats = K // wf.shape[0] + 1
            wf = wf.repeat(repeats)[:K]
        waveforms.append(wf)

    result_batch = {
        "data_object": torch.stack(waveforms),
        "labels": torch.tensor([elem["labels"] for elem in dataset_items]).long(),
    }
    return result_batch
