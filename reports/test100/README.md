# Test-100 Evaluation

This folder contains evaluation reports for the 100-video Test-100 set.

## Hugging Face Dataset

- Public dataset URL: https://huggingface.co/datasets/Seoyoung07/korean-sign-word-classifier-mediapipe-test-100
- Upload commit: `5dc76d221db9b74cc719cbbfb7528c7b5ec6a56d`
- AI tool used: Codex

## Latest Summary

- Latest report: `test100_report_20260606_171322_ko.md`
- Model: `Seoyoung07/korean-sign-word-classifier-mediapipe`
- Model snapshot used for evaluation: `214c0d87294695f4c79086440404bfb72fa7c5bc`
- Checkpoint epoch: 41
- Checkpoint validation accuracy: 0.8962213765938981
- Total videos: 100
- Top-1 correct: 60 / 100 (60.00%)
- Top-5 contains expected: 71 / 100 (71.00%)
- Inference errors: 0

## Files

- `test100_report_20260606_171322_ko.md`: Korean summary and full result table for the latest evaluation.
- `test100_report_20260606_171322.md`: English full result table for the latest evaluation.
- `test100_results_20260606_171322.csv`: Machine-readable CSV results.
- `test100_results_20260606_171322.json`: Machine-readable JSON results.
- Older `20260604` files are retained for comparison with the earlier checkpoint.

## Label Note

The original filename typo `꺠끗하다.mp4` is preserved in the uploaded Hugging Face dataset, while the canonical label is recorded as `깨끗하다`.
