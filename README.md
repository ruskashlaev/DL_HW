# ASVspoof2019 LA Anti-Spoofing (LCNN)

A countermeasure (CM) model for detecting spoofed/synthetic speech on the
logical access (LA) track of [ASVspoof2019](https://www.asvspoof.org/index2019.html).
The model is an LCNN (Light CNN) on top of a mel-spectrogram, trained and configured
via [Hydra](https://hydra.cc/); the evaluation metric is EER.

This repository is based on [pytorch_project_template](https://github.com/Blinorot/pytorch_project_template)
(HSE DLA course).

## Task

Binary audio classification: `bonafide` (genuine speech) vs `spoof` (synthesized/converted
speech). The metric is **Equal Error Rate (EER)** — the threshold at which `FAR == FRR`.

Key property of the ASVspoof2019 LA protocol: `dev` contains **the same attack algorithms**
as `train` (A01–A06), while `eval` contains **different, unseen** attacks (A07–A19).
Because of this, EER on `dev` barely reflects the model's real generalization ability —
`eval` is the number that matters.

## Installation

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

## Data

Expected layout (ASVspoof2019 LA CM protocols):

```
data/asvspoof/
├── train.txt          # speaker utt_id - attack_id bonafide|spoof
├── dev.txt
├── eval.txt
├── train/flac/*.flac
├── dev/flac/*.flac
└── eval/flac/*.flac
```

On first run, [`ASVSpoofDataset`](src/datasets/asvspoof.py) builds and caches
`data/asvspoof/<split>/index.json` from the corresponding `.txt` protocol. If the
protocol changes but `index.json` already exists, it is **not rebuilt automatically** —
delete the file manually.

## Training

```bash
python train.py -cn=asvspoof
```

Config: [`src/configs/asvspoof.yaml`](src/configs/asvspoof.yaml).

- model: [LCNN](src/model/asvspoof_lcnn.py), input — 80 mel channels
- batch transform: `MelSpectrogram(n_fft=512, hop=160, n_mels=80)` + `AmplitudeToDB`
  ([src/configs/transforms/batch_transforms/asvspoof.yaml](src/configs/transforms/batch_transforms/asvspoof.yaml))
- optimizer: Adam(lr=3e-4) + StepLR
- metric: [EER](src/metrics/eer.py) (class 0 = bonafide, class 1 = spoof)
- checkpoint selection — by `min eval_EER` (not `val_EER`: dev shares attacks with
  train, so its EER quickly collapses to 0 and stops being informative)

Checkpoints and logs are saved to `saved/<writer.run_name>/`
(run name = `writer.run_name`, see [src/configs/writer/wandb.yaml](src/configs/writer/wandb.yaml)).

### Smoke test

A run on a small slice of `dev` (100 samples) to make sure the pipeline runs at all
and the model can overfit a tiny set:

```bash
python train.py -cn=asvspoof_smoketest
```

## Exporting scores (for grading/submission)

`inference.py` + `src/configs/inference.yaml` in this repo are still set up for the
template's generic example (`baseline`/`example`), not for ASVspoof. Score generation
for this task uses a separate script, [`export_scores.py`](export_scores.py):

```bash
venv\Scripts\python.exe export_scores.py
```

Before running, edit in the file:

- `STUDENT_NAME` — used for the output filename and must match what the grading
  script expects;
- `CHECKPOINT_PATH` — path to the checkpoint in `saved/<run_name>/`.

The script runs the `eval` split through the model using the config saved alongside
the checkpoint (`saved/<run_name>/config.yaml`), and writes `<STUDENT_NAME>.csv` in
`key,score` format **with no header**, where score = `logits[:, 0] - logits[:, 1]`
(higher → more bonafide-like). This sign convention is required — the grading script
expects exactly this direction.

## Results

EER is computed on the full eval split (unseen attacks A07–A19). After 15-20 epochs
with the current config, a typical result is **EER ≈ 6.3–6.5%** on eval, while val
(dev) EER drops to 0 by epoch 3-5 — this is expected (see the Task section) and is
not a sign of overfitting or data leakage, nor of the model's actual final quality.

## Repository structure

```
src/
├── configs/          # Hydra configs (model, datasets, transforms, trainer...)
├── datasets/         # ASVSpoofDataset, collate_fn (crop/pad to a fixed length)
├── model/             # LCNN
├── metrics/           # EER (src/metrics/eer.py, calculate_eer.py)
├── trainer/           # BaseTrainer / Trainer / Inferencer
└── logger/            # WandB / Comet ML
train.py               # training
inference.py            # generic inference from the template
export_scores.py        # ASVspoof score export for submission/grading
```

## License

[MIT](/LICENSE), based on [pytorch_project_template](https://github.com/Blinorot/pytorch_project_template).
