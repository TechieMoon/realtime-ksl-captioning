# Self-made-60 Evaluation

This folder contains the evaluation report for 60 newly recorded self-made videos.

## Summary

- Input folder used for evaluation: `/home/moon/Downloads/self-made-60`
- Original recording names were timestamp-based.
- Files were renamed to word labels before evaluation.
- Rename mapping: `rename_mapping_20260606_193217.csv`
- Model: `Seoyoung07/korean-sign-word-classifier-mediapipe`
- Model snapshot: `214c0d87294695f4c79086440404bfb72fa7c5bc`
- Checkpoint epoch: 41
- Checkpoint validation accuracy: 0.8962213765938981
- Device: `cuda`
- Total videos: 60
- Top-1 correct: 5 / 60 (8.33%)
- Top-5 contains expected: 7 / 60 (11.67%)
- Inference errors: 0

## Files

- `self_made_60_report_20260606_193217_ko.md`: Korean summary and full result table.
- `self_made_60_report_20260606_193217.md`: English full result table.
- `self_made_60_results_20260606_193217.csv`: Machine-readable CSV results.
- `self_made_60_results_20260606_193217.json`: Machine-readable JSON results.
- `rename_mapping_20260606_193217.csv`: Original timestamp filename to word filename mapping.

## Correct Words

Top-1 correct words:

```text
갈등, 골짜기, 머리, 먹구름, 바람
```

Top-1 failed but Top-5 contained the expected word:

```text
교제, 머리카락
```
