# Hugging Face Dataset Uploads

This document summarizes the public Hugging Face datasets uploaded for evaluation and TA review.

## AI Tool Used

- AI tool used: Codex

## Related Model

- Model repo: `Seoyoung07/korean-sign-word-classifier-mediapipe`
- Model URL: https://huggingface.co/Seoyoung07/korean-sign-word-classifier-mediapipe
- Model checkpoint used for evaluation: epoch 41
- Model validation accuracy recorded in checkpoint: 0.8962213765938981

## Public Dataset Repositories

| Dataset | Hugging Face URL | Visibility | Commit | Videos | Top-1 | Top-5 | Errors |
|---|---|---|---|---:|---:|---:|---:|
| Test-100 | https://huggingface.co/datasets/Seoyoung07/korean-sign-word-classifier-mediapipe-test-100 | Public | `5dc76d221db9b74cc719cbbfb7528c7b5ec6a56d` | 100 | 60 / 100 | 71 / 100 | 0 |
| Self-made-60 | https://huggingface.co/datasets/Seoyoung07/korean-sign-word-classifier-mediapipe-self-made-60 | Public | `9e11d935526a54994d8f33e56649213fafa02057` | 60 | 5 / 60 | 7 / 60 | 0 |

## Dataset File Layout

Each Hugging Face dataset repo contains:

```text
README.md
metadata.csv
videos/*.mp4
evaluation/*
```

`metadata.csv` includes:

- `file_name`: relative video path inside the dataset repo.
- `label`: canonical expected word label.
- `label_raw`: original label before normalization, when applicable.
- `label_alias_applied`: whether a filename or label alias was applied.
- `predicted`: model Top-1 prediction.
- `confidence`: Top-1 probability.
- `top1_correct`: whether Top-1 matched the expected label.
- `top5_contains_expected`: whether the expected label appeared in Top-5.
- `expected_in_vocab`: whether the expected label exists in the model vocabulary.
- `input_frames`, `effective_frames`: frames used by the inference wrapper.
- `source`: dataset source identifier.
- `ai_tool_used`: `Codex`.

## Test-100 Dataset

Hugging Face:

```text
https://huggingface.co/datasets/Seoyoung07/korean-sign-word-classifier-mediapipe-test-100
```

GitHub report:

```text
reports/test100/test100_report_20260606_171322_ko.md
```

Notes:

- Contains 100 isolated Korean sign-language word videos.
- The original filename typo `꺠끗하다.mp4` is preserved in `videos/`.
- The canonical label is recorded as `깨끗하다` in `metadata.csv`.
- Evaluation result: Top-1 60 / 100, Top-5 71 / 100, errors 0.

## Self-made-60 Dataset

Hugging Face:

```text
https://huggingface.co/datasets/Seoyoung07/korean-sign-word-classifier-mediapipe-self-made-60
```

GitHub report:

```text
reports/self-made-60/self_made_60_report_20260606_193217_ko.md
```

Notes:

- Contains 60 newly recorded self-made isolated word videos.
- These words were selected from the Test-100 words that were Top-1 correct in the prior evaluation.
- Original files were timestamp-named recordings.
- Files were sorted chronologically and renamed to canonical word labels before evaluation.
- The full filename mapping is in:

```text
reports/self-made-60/rename_mapping_20260606_193217.csv
```

- Evaluation result: Top-1 5 / 60, Top-5 7 / 60, errors 0.

## Download Examples

Using Python:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Seoyoung07/korean-sign-word-classifier-mediapipe-test-100",
    repo_type="dataset",
    local_dir="hf-test-100",
)

snapshot_download(
    repo_id="Seoyoung07/korean-sign-word-classifier-mediapipe-self-made-60",
    repo_type="dataset",
    local_dir="hf-self-made-60",
)
```

Using CLI:

```bash
huggingface-cli download Seoyoung07/korean-sign-word-classifier-mediapipe-test-100 \
  --repo-type dataset \
  --local-dir hf-test-100

huggingface-cli download Seoyoung07/korean-sign-word-classifier-mediapipe-self-made-60 \
  --repo-type dataset \
  --local-dir hf-self-made-60
```

## Verification

The upload was verified without using a token:

- Test-100 visibility: public
- Test-100 remote video count: 100
- Self-made-60 visibility: public
- Self-made-60 remote video count: 60
- Both dataset cards contain `AI tool used: Codex`.
