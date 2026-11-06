# FastSpeech 2 text-to-speech for English and Telugu

Two separate FastSpeech 2 models, one for English (Svarah) and one for Telugu (IndicTTS Telugu), built on the PyTorch
implementation [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (MIT licence, see `fastspeech2/LICENSE`).
The full write-up is in `Report/Technical_Report.pdf`.

| | English | Telugu |
|---|---|---|
| Data | [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (Indian-accented English, 117 speakers) | [SPRINGLab/IndicTTS_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) (2 studio voices) |
| Selected for training / validation | 4.99 h (3452 clips) / 1.11 h (748 clips) | 5.00 h (2245 clips) / 1.00 h (459 clips) |
| Used after alignment and silence trimming | 3.69 h / 0.82 h | 3.57 h / 0.72 h |
| Speakers in the model | 103 (speaker groups built from the metadata) | 2 |
| Final checkpoint | step 15000 | step 15000 |

## Folder layout

```
English/  Telugu/       train.py  preprocess.py  inference.py  config.yaml  test_sentences.json  checkpoints/
Results/English|Telugu  training_loss.png  validation_loss.png  test_generated_audios/  variance_adaptor_plots/{duration,pitch,energy}/
Results/comparison/     version 1 vs version 2 loss curves, prosody figure
Results_v1/             results of the first training run (baseline), same layout
Report/Technical_Report.pdf
common/                 code shared by both languages (data selection, text front ends, training loop, evaluation, plots)
fastspeech2/            model code of the original repository, with the patches listed below
notebooks/v1_baseline/  Colab notebooks of the first run (30000 steps)
notebooks/v2_final/     Colab notebooks of the final run (15000 steps); these produced the files in Results/
tests/smoke_test.py     end-to-end test on synthetic speech (CPU, a few minutes)
```

`English/*.py` and `Telugu/*.py` are thin wrappers; the logic is in `common/pipeline.py`. What differs per language is
`config.yaml`, the text front end (`common/english_frontend.py`, `common/telugu_frontend.py`) and the dataset.

Each `Results/<Language>/` folder holds the outputs of the final checkpoint. `extra/` inside it contains the checkpoint
selection scores, the outputs of the checkpoint that the objective score preferred, and the normalisation / dictionary
ablations. `variance_adaptor_values.json` lists the predicted duration, pitch and energy of every phone of every test sentence.

## Checkpoints

Weights are stored in half precision (about 70 MB per file) so that they fit on GitHub without Git LFS.

| File | Content |
|---|---|
| `English/checkpoints/selected_model_fp16.pth` | final English model (step 15000) |
| `English/checkpoints/best_model_fp16.pth` | lowest validation loss of the final run (step 4000) |
| `English/checkpoints/v1_baseline/last_model_fp16.pth` | first run, last step (30000) |
| `Telugu/checkpoints/selected_model_fp16.pth` | final Telugu model (step 15000) |
| `Telugu/checkpoints/best_model_fp16.pth` | lowest validation loss of the final run (step 1000) |
| `Telugu/checkpoints/v1_baseline/last_model_fp16.pth` | first run, step 8000 (training stopped by the GPU quota) |

The training snapshots (one every 1000 steps) and the resumable training state stay on Google Drive and are not part of the repository.

## Reproduce

```bash
pip install -r requirements.txt
# forced aligner, in a separate environment:
micromamba create -n mfa -c conda-forge python=3.11 montreal-forced-aligner
export HF_TOKEN=hf_xxx            # Hugging Face read token (Svarah is a gated dataset)

# English
python English/preprocess.py --stage all        # select -> lexicon -> align -> features
python English/train.py                         # resumable: run again to continue; --max_minutes N stops cleanly
python English/inference.py --checkpoint English/checkpoints/selected_model_fp16.pth

# Telugu
python Telugu/preprocess.py --stage all
python Telugu/train.py
python Telugu/inference.py --checkpoint Telugu/checkpoints/selected_model_fp16.pth

# score every training snapshot on held-out validation sentences
python common/evaluate_snapshots.py --lang_dir English --ckpt_dir <work>/ckpt --out_dir Results/English/extra/checkpoint_selection
```

Large temporary files go to `<language>/work/` (or `$FS2_WORK`) and are git-ignored. Hyper-parameters, data selection
settings and paths are all in the language's `config.yaml`. `config_v1.yaml` is the configuration of the first run.
Other inference options: `--g2p_mode lexicon` (English: plain dictionary lookup), `--no_normalize` (skip number and time
normalisation), `--all_speakers` (also synthesise with every other speaker).

## Pipeline in short

1. `select`: download the dataset, keep clips of 1 to 12 s (1 to 15 s for Telugu), drop speakers with less than 60 s of audio,
   take the same fraction of every speaker's audio with a fixed seed, split by utterance into train and validation, write
   22.05 kHz wav files with normalised transcripts.
2. `lexicon`: English uses the LibriSpeech lexicon plus `g2p_en` for unknown words; Telugu uses a rule-based
   script-to-phone converter.
3. `align`: Montreal Forced Aligner is trained on the selected data and writes phone-level alignments.
4. `features`: mel-spectrogram, phoneme-level pitch and energy, and durations from the alignments. Pitch and energy
   statistics are computed on the training files only.
5. `train`: FastSpeech 2 with the original loss (mel, postnet mel, pitch, energy, duration).
6. `inference`: text, then phones, then mel-spectrogram, then waveform with the pretrained HiFi-GAN (universal) vocoder.

## Changes to the original code (marked `[PATCH]` in the source)

* `audio/stft.py`, `audio/audio_processing.py`: keyword arguments for librosa 0.10 and later, no hard-coded `.cuda()`.
* `text/symbols.py`: 47 Telugu phones added, Mandarin symbols removed.
* `utils/model.py`: vocoder weights unzipped on demand, `map_location`, only HiFi-GAN kept.
* `utils/tools.py`, `model/loss.py`: float32 casts for pitch, energy and mel targets (the feature files are stored as float64).
* `model/optimizer.py`: AdamW (decoupled weight decay) instead of Adam.
* `preprocessor/preprocessor.py`: explicit train / validation split, statistics from training files only, fix for a missing
  alignment file, explicit sampling rate, utf-8.
* The original `train.py`, `synthesize.py`, `evaluate.py` and `prepare_align.py` are replaced by `common/*`: resumable
  training with CSV loss logs, a snapshot every 1000 steps, variance-adaptor plots, and dataset-specific preparation.
  Model, loss and dataset classes are unchanged.
* Telugu support: `common/telugu_frontend.py` (number, time and ordinal expansion, grapheme-to-phone rules), Telugu
  symbols, `Telugu/config.yaml`.

## Training runs

* Version 1 (`config_v1.yaml`, `Results_v1/`): 30000 steps for English; the Telugu run was stopped at about step 8700 when
  the GPU quota ran out. The lowest validation loss came very early in both languages (English step 4000, Telugu step 1500)
  and the audio kept improving after that, so validation loss was not a reliable way to choose a checkpoint.
* Version 2 (`config.yaml`, `Results/`): 15000 steps, learning rate reduced earlier, AdamW with weight decay 0.01, a
  snapshot every 1000 steps, and every snapshot scored on held-out validation sentences (`common/evaluate_snapshots.py`).
  The final checkpoint of each language was chosen after listening to the candidates.

## Notes and limitations

* The assignment PDF prints E3 as "rhyth m" and T3 with a gap inside the word అడవిలో; we synthesise "rhythm" and
  అడవిలో. The printed text is kept in `test_sentences.json` (`as_printed`).
* Svarah is a speech recognition benchmark (conversational, many speakers, mixed recording conditions), not a speech
  synthesis corpus, so the English model is much weaker than the Telugu model trained on studio recordings.
* The Telugu number words and pronunciation rules were written from standard usage and were not checked by a native speaker.
* The test speaker is the one with the most training data (English `spk013`, Telugu speaker 1).
