import torchaudio
from tqdm.auto import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class ASVSpoofDataset(BaseDataset):
    def __init__(self, name="train", *args, **kwargs):
        index_path = ROOT_PATH / "data" / "asvspoof" / name / "index.json"

        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, name):
        index = []
        data_path = ROOT_PATH / "data" / "asvspoof" / name
        data_path.mkdir(exist_ok=True, parents=True)

        with open(f"{data_path}.txt", "r", encoding="utf-8") as f:
            for i in tqdm(f):
                record = i.strip().split()
                record_path = data_path / f"{record[1]}.flac"
                record_label = record[-1]
                index.append({"path": str(record_path), "label": record_label})

        write_json(index, str(data_path / "index.json"))

        return index

    def load_object(self, path):
        waveform, sr = torchaudio.load(path)
        return waveform
