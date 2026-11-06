# FastSpeech 2 Text-to-Speech for English and Telugu

This repository is my submission for the assignment "Implementation of FastSpeech 2-Based Text-to-Speech Synthesis" (Speech Processing Lab, LTRC, IIIT Hyderabad). The task is to train two separate FastSpeech 2 models, one for English and one for Telugu, using about 5 hours of training audio and 1 hour of validation audio for each language, and to generate speech for five test sentences per language.

The English model is trained on Svarah (Indian-accented English) and the Telugu model on the IndicTTS Telugu corpus. Both start from the PyTorch FastSpeech 2 implementation by ming024 and use the pretrained HiFi-GAN vocoder from that repository. The Telugu text front end, the data selection, the training loop and the evaluation scripts are my own additions.

Everything is inside the `project/` folder. The written report is `project/Report/Technical_Report.pdf`.

## What is in the repository

```
project/
  English/               train.py, preprocess.py, inference.py, config.yaml, checkpoints/
  Telugu/                train.py, preprocess.py, inference.py, config.yaml, checkpoints/
  Results/
    English/             loss curves, generated test audio, variance adaptor plots
    Telugu/              loss curves, generated test audio, variance adaptor plots
    comparison/          loss and prosody comparison between the two training runs
  Results_v1/            results of the first training run, kept as a baseline
  Report/                Technical_Report.pdf (and an editable .docx)
  common/                code shared by both languages
  fastspeech2/           model code from the reference repository, with my patches
  notebooks/             Colab notebooks used to run the experiments (v1_baseline, v2_final)
  tests/                 a small end-to-end test on synthetic data
  README.md              detailed usage notes, checkpoint table and list of code changes
  requirements.txt
```

## Where to find each deliverable

| Required item | Location |
|---|---|
| Training and validation code for each language | `project/English/` and `project/Telugu/` (scripts call the shared code in `project/common/`) |
| Configuration files | `project/English/config.yaml`, `project/Telugu/config.yaml` |
| Model checkpoints | `project/English/checkpoints/`, `project/Telugu/checkpoints/` |
| Training and validation loss plots | `project/Results/<Language>/training_loss.png`, `validation_loss.png` |
| Generated speech for the five test sentences | `project/Results/<Language>/test_generated_audios/` |
| Duration, pitch and energy plots | `project/Results/<Language>/variance_adaptor_plots/duration`, `pitch`, `energy` |
| Technical report | `project/Report/Technical_Report.pdf` |
| Dependencies | `project/requirements.txt` |

## Short summary of what was done

1. Data selection: the data is subsampled evenly across speakers to about 5 hours for training and 1 hour for validation per language, with no utterance shared between the two sets.
2. Text front end: English uses the LibriSpeech lexicon with a neural grapheme-to-phoneme model for unknown words, and a normaliser for numbers and times. Telugu uses a rule-based converter from script to 47 phones, with number, time and ordinal expansion.
3. Alignment and features: phone durations come from the Montreal Forced Aligner. Mel-spectrograms, phone-level pitch and energy are extracted from the audio.
4. Training: 35.1 million parameter FastSpeech 2 models, trained on a Colab T4 GPU. A first run (30000 steps for English, about 8700 for Telugu) was followed by a second, shorter run (15000 steps) with AdamW weight decay and a checkpoint snapshot every 1000 steps.
5. Checkpoint selection: the lowest validation loss came very early in both languages, and the audio at that point was not the best. Each snapshot was therefore scored on held-out sentences and the final checkpoint (step 15000 for both languages) was chosen by listening.
6. Evaluation: loss curves, duration, pitch and energy plots for every test sentence, and a comparison of English and Telugu, all discussed in the report.

## Results in brief

The Telugu model gives clearer and more natural speech than the English model. The English data comes from a speech recognition benchmark with many speakers and varied recording conditions, so the English output follows the right rhythm and length but is hard to understand and has almost flat pitch. In both languages the pitch and energy predictors overfit, while duration prediction is good. The report gives the numbers and the reasoning.

## Running the code

The full command list is in `project/README.md`. In short:

```bash
pip install -r project/requirements.txt
cd project
python English/preprocess.py --stage all
python English/train.py
python English/inference.py --checkpoint English/checkpoints/selected_model_fp16.pth
```

The same three commands work for `Telugu/`. The Svarah dataset is gated on Hugging Face, so a read token is needed for the English data. The Montreal Forced Aligner has to be installed separately, as described in `project/README.md`.

## Notes

* The assignment prints "rhyth m" (English) and a gap inside the Telugu word for "in the forest" in the test sentences. I synthesised the intended words. The text as printed is stored in `test_sentences.json` in each language folder.
* The Telugu pronunciation rules and number words were not checked by a native speaker, so the Telugu evaluation relies on listening and signal measurements.
* The checkpoints are stored in half precision (about 70 MB each), so Git LFS is not needed.
* The first run is kept under `Results_v1/` and `notebooks/v1_baseline/` for comparison. The final results are in `Results/` and `notebooks/v2_final/`.

## Credits

FastSpeech 2: Ren et al., ICLR 2021. PyTorch implementation: ming024/FastSpeech2 (MIT licence). Vocoder: HiFi-GAN (Kong et al., 2020). Alignment: Montreal Forced Aligner. Datasets: Svarah (AI4Bharat) and IndicTTS Telugu (SPRING Lab, IIT Madras).
