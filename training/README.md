# MediaPipe MVP Training

This folder contains the reproducible training pipeline for the one-person real-time KSL MVP.

## What It Trains

The script trains a lightweight classifier over MediaPipe Holistic keypoint sequences. It is designed to prove that the server can receive webcam video, run real-time inference, and return Korean word captions.

It is not a production-quality full Korean Sign Language translation model.

Current default vocabulary:

```text
수어, 좋다, 감사, 괜찮다, 싫다, 이해, 부탁, 모르다, 맞다, 힘
```

## Required Local Data

Do not commit AIHub zip files to GitHub.

Place downloaded AIHub files under the repository root with this structure:

```text
수어 영상/
  1.Training/
    [라벨]01_real_word_morpheme.zip
    [원천]01_real_word_video.zip
  2.Validation/
    [라벨]01_real_word_morpheme.zip
    [원천]01_real_word_video.zip
```

For the first MVP run, the minimum useful files are:

```text
수어 영상/1.Training/[라벨]01_real_word_morpheme.zip
수어 영상/1.Training/[원천]01_real_word_video.zip
수어 영상/2.Validation/[라벨]01_real_word_morpheme.zip
수어 영상/2.Validation/[원천]01_real_word_video.zip
```

The script reads directly from zip files, so full extraction is not required.

## Install

From the repository root:

```powershell
python -m pip install -r server/requirements-dev.txt
python -m pip install -r ai_model/requirements.txt
```

## Train And Evaluate

Run:

```powershell
python training\train_mediapipe_mvp.py --max-per-label 20 --sequence-length 16 --angles F D U --labels "수어,좋다,감사,괜찮다,싫다,이해,부탁,모르다,맞다,힘"
```

The script automatically:

- reads matching videos and morpheme labels from local AIHub zip files
- extracts MediaPipe Holistic keypoints
- trains the classifier
- prints validation accuracy and a per-class classification report
- saves `ai_model/mediapipe_mvp.joblib`

The saved model artifact is ignored by Git and should be uploaded to Hugging Face instead.

## Upload To Hugging Face

After training:

```powershell
hf auth login
hf upload TechieMoon/realtime-ksl-captioning-mediapipe-mvp ai_model\mediapipe_mvp.joblib mediapipe_mvp.joblib --type model --commit-message "Update MediaPipe MVP model"
hf upload TechieMoon/realtime-ksl-captioning-mediapipe-mvp ai_model\README.md README.md --type model --commit-message "Update model card"
```

## Data Growth Plan

Use this order:

```text
1. Word data only: real_word
2. Real captured video only: REAL
3. Start with frontal angle F
4. Add U/D angles for robustness
5. Add L/R angles only if demo camera angle varies
6. Add more [원천]XX_real_word_video.zip chunks as storage/time allow
7. Keep sentence SEN, synthetic SYN, and crowd data out of the first MVP
```

Recommended stages:

```text
Stage 1: 01_real_word train + 01_real_word validation, 10 words
Stage 2: 01-04_real_word train + validation, 10-20 words
Stage 3: 01-16_real_word train + validation, 20-50 words
Stage 4: all 01-32_real_word train chunks only if storage and training time are acceptable
```

For the current project goal, do not use sentence data until word-level recognition is stable.
