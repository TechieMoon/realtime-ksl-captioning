# Sign Language Model Structure

This folder contains the model-side implementation for a Korean sign-language
subtitle pipeline.

```text
raw keypoints [B, T, K, 4]
  -> KeypointPreprocessor
  -> KeypointToGlossCTC
  -> greedy_ctc_decode / StableGlossBuffer
  -> Gloss2TextTransformer
  -> Korean subtitle tokens
```

## Files

- `config.py`: model and keypoint-layout configuration.
- `keypoint_extractor.py`: MediaPipe Holistic mp4/frame extractor and 543-to-115 keypoint selector.
- `preprocessing.py`: body-centered normalization, velocity, acceleration, wrist-relative features.
- `keypoint_ctc.py`: spatial keypoint encoder + temporal Transformer + CTC gloss head.
- `ctc_decoder.py`: greedy CTC decoding and realtime gloss stabilization.
- `gloss2text.py`: Transformer encoder-decoder for gloss-to-Korean-text generation.
- `pipeline.py`: wrapper that connects video extraction, CTC decoding, and optional gloss-to-text generation.
- `word_dataset.py`: matches mp4 files with morpheme JSON labels and reads keypoint caches.
- `word_classifier.py`: isolated-word keypoint classifier for the current word-clip dataset.
- `prepare_keypoints.py`: extracts and saves `.npy` keypoint caches.
- `train_word_classifier.py`: trains the isolated-word classifier with cross entropy.

## Default Keypoint Layout

The default layout assumes this order:

```text
upper body pose: 13 points
left hand:       21 points
right hand:      21 points
selected face:   60 points
total:          115 points
```

Raw frame input must be:

```text
[B, T, K, 4] = [batch, frames, keypoints, x/y/z/confidence]
```

`KeypointPreprocessor` expands it to:

```text
[B, T, K, 16]
```

The 16 features are:

```text
normalized xyz
confidence
velocity xyz
acceleration xyz
relative xyz to left wrist
relative xyz to right wrist
```

## Holistic Extraction

`HolisticKeypointExtractor` uses `mediapipe.solutions.holistic.Holistic`:

```text
MediaPipe Holistic output:
33 pose + 468 face + 21 left hand + 21 right hand = 543 landmarks

Model input:
13 upper-body pose + 21 left hand + 21 right hand + 60 selected face = 115 landmarks
```

The selected output is saved or returned as:

```text
[T, 115, 4] = [frames, keypoints, x/y/z/confidence]
```

Example:

```python
from model.keypoint_extractor import HolisticKeypointExtractor

extractor = HolisticKeypointExtractor()
keypoints = extractor.extract_video("sample.mp4")
extractor.extract_video_to_npy("sample.mp4", "cache/sample.npy")
```

## Example

```python
import torch

from model import (
    Gloss2TextConfig,
    Gloss2TextTransformer,
    KeypointCTCConfig,
    KeypointToGlossCTC,
    StableGlossBuffer,
    greedy_ctc_decode,
)

ctc_config = KeypointCTCConfig(
    num_keypoints=115,
    input_features=16,
    gloss_vocab_size=1000,
)
keypoint_model = KeypointToGlossCTC(ctc_config)

raw_keypoints = torch.randn(2, 64, 115, 4)
lengths = torch.tensor([64, 58])

out = keypoint_model.forward_raw(raw_keypoints, lengths=lengths)
decoded = greedy_ctc_decode(out["logits"], blank_id=ctc_config.blank_id, lengths=lengths)

buffer = StableGlossBuffer(history_size=4, min_repeats=2)
stable = buffer.update(decoded[0])

g2t_config = Gloss2TextConfig(gloss_vocab_size=1000, text_vocab_size=8000)
g2t_model = Gloss2TextTransformer(g2t_config)

gloss_tokens = torch.tensor([[1, *stable["committed"], 2]])
subtitle_tokens = g2t_model.greedy_generate(gloss_tokens)
```

## Training

Keypoint-to-gloss training uses CTC:

```python
out = keypoint_model.forward_raw(raw_keypoints, lengths=input_lengths)
loss = keypoint_model.ctc_loss(
    out["log_probs"],
    target_gloss_ids,
    out["lengths"],
    target_lengths,
)
```

Gloss-to-text training uses teacher forcing:

```python
logits = g2t_model(gloss_tokens, text_input_tokens)
loss = cross_entropy(logits[:, :-1], text_target_tokens[:, 1:])
```

The two models should normally be trained separately first, then validated as a
full pipeline.

## Current Word-Clip Training Flow

The provided dataset is isolated word clips:

```text
mp4 clip -> one Korean word label
```

So the first training target should be word classification, not CTC:

```text
mp4
  -> HolisticLandmarker
  -> [T, 115, 4] keypoint cache
  -> WordKeypointClassifier
  -> CrossEntropyLoss(word_id)
```

Create keypoint caches:

```powershell
python -m model.prepare_keypoints `
  --split-root "1.Training" `
  --cache-root keypoint_cache `
  --manifest-out manifests/train.json `
  --task-model-path model/assets/holistic_landmarker.task `
  --skip-existing

python -m model.prepare_keypoints `
  --split-root "2.Validation" `
  --cache-root keypoint_cache `
  --manifest-out manifests/val.json `
  --task-model-path model/assets/holistic_landmarker.task `
  --skip-existing
```

Train on GPU:

```powershell
python -m model.train_word_classifier `
  --train-manifest manifests/train.json `
  --val-manifest manifests/val.json `
  --output-dir checkpoints/word_classifier `
  --epochs 20 `
  --batch-size 16 `
  --max-frames 64
```

The best checkpoint is saved to:

```text
checkpoints/word_classifier/best.pt
```
