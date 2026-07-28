import random

import torch


def collate_fn(dataset_items: list[dict]):
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
