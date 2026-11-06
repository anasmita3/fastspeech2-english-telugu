# FastSpeech 2 text-to-speech for English and Telugu

This folder contains the code, configuration files, trained checkpoints, results and report for two separate FastSpeech 2 models: one for English trained on Svarah, and one for Telugu trained on IndicTTS Telugu. The work was done for the assignment "Implementation of FastSpeech 2-Based Text-to-Speech Synthesis" of the Speech Processing Lab at LTRC, IIIT Hyderabad.

The models are built on the PyTorch implementation [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (MIT licence, copy in `fastspeech2/LICENSE`). Speech is produced from the predicted mel-spectrograms with the pretrained HiFi-GAN (universal) vocoder that ships with that repository; the vocoder is not retrained. The full write-up, with the analysis of the loss curves, the generated speech and the variance adaptor plots, is in `Report/Technical_Report.pdf`.

| | English | Telugu |
|---|---|---|
| Data | [ai4bharat/Svarah](https://huggingface.co/datasets/ai4bharat/Svarah), Indian-accented English, 117 speakers | [SPRINGLab/IndicTTS_Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu), 2 studio voices |
| Selected for training / validation | 4.99 h (3452 clips) / 1.11 h (748 clips) | 5.00 h (2245 clips) / 1.00 h (459 clips) |
| Used after alignment and silence trimming | 3.69 h / 0.82 h | 3.57 h / 0.72 h |
| Speakers in the model | 103 (speaker groups built from the metadata) | 2 |
| Parameters | 35.1 million | 35.1 million |
| Final checkpoint | step 15000 | step 15000 |

## Folder layout

```
English/                 train.py, preprocess.py, inference.py, config.yaml, test_sentences.json, checkpoints/
Telugu/                  the same files for Telugu
Results/
  English/               training_loss.png, validation_loss.png, test_generated_audios/, variance_adaptor_plots/
  Telugu/                the same for Telugu
  comparison/            loss curves of the two training runs side by side, prosody comparison
Results_v1/              results of the first training run (baseline), same layout as Results/
Report/                  Technical_Report.pdf and an editable .docx copy
common/                  code shared by both languages
fastspeech2/             model code of the reference repository, with my patches
notebooks/v1_baseline/   Colab notebooks of the first run
notebooks/v2_final/      Colab notebooks of the final run; these produced the files in Results/
tests/smoke_test.py      end-to-end test on synthetic speech (CPU, a few minutes)
requirements.txt
```

The scripts in `English/` and `Telugu/` are short wrappers; the logic lives in `common/pipeline.py`. What differs between the two languages is the `config.yaml`, the text front end (`common/english_frontend.py` and `common/telugu_frontend.py`) and the dataset.

Each `Results/<Language>/` folder holds the outputs of the final checkpoint:

* `training_loss.png` and `validation_loss.png`: total loss and its five parts (mel, post-net mel, pitch, energy, duration). The CSV files with the logged values are next to them.
* `test_generated_audios/`: one wav file for each of the five test sentences.
* `variance_adaptor_plots/duration`, `pitch`, `energy`: one plot per test sentence. The numbers behind the plots are in `variance_adaptor_values.json`.
* `extra/`: the checkpoint selection scores, outputs of the checkpoint that the objective score preferred, and two small ablations (plain dictionary lookup instead of context-aware pronunciation, and no number normalisation).

## Checkpoints

Weights are stored in half precision, about 70 MB per file, so they fit on GitHub without Git LFS.

| File | Content |
|---|---|
| `English/checkpoints/selected_model_fp16.pth` | final English model (step 15000) |
| `English/checkpoints/best_model_fp16.pth` | lowest validation loss of the final run (step 4000) |
| `English/checkpoints/v1_baseline/last_model_fp16.pth` | first run, last step (30000) |
| `Telugu/checkpoints/selected_model_fp16.pth` | final Telugu model (step 15000) |
| `Telugu/checkpoints/best_model_fp16.pth` | lowest validation loss of the final run (step 1000) |
| `Telugu/checkpoints/v1_baseline/last_model_fp16.pth` | first run, step 8000 (training stopped when the GPU quota ran out) |

The training snapshots (one every 1000 steps) and the resumable training state stayed on Google Drive and are not part of the repository.

## How the pipeline works

1. **select** (`common/data_select.py`): download the dataset, keep clips of 1 to 12 s (1 to 15 s for Telugu), drop speakers with less than 60 s of audio, take the same fraction of every speaker's audio with a fixed seed, split by utterance into training and validation sets, and write 22.05 kHz wav files with normalised transcripts. The split is saved in `split.json` so that every later step uses the same files.
2. **lexicon**: English uses the LibriSpeech lexicon and `g2p_en` for words that are not in it. Telugu uses a rule-based script-to-phone converter, because Telugu spelling is close to phonetic.
3. **align**: the Montreal Forced Aligner is trained on the selected data and writes phone-level alignments.
4. **features**: mel-spectrograms, phoneme-level pitch and energy, and durations from the alignments. Pitch and energy statistics are computed on the training files only.
5. **train** (`common/train_loop.py`): FastSpeech 2 with the original loss (mel, post-net mel, pitch, energy, duration). Training can be resumed, loss values are written to CSV files, and a snapshot is saved every 1000 steps.
6. **inference** (`common/synth.py`): text to phones, phones to mel-spectrogram, mel-spectrogram to waveform with HiFi-GAN. The variance adaptor plots are produced in the same step.

## Reproducing the results

```bash
pip install -r requirements.txt
# forced aligner, in a separate environment:
micromamba create -n mfa -c conda-forge python=3.11 montreal-forced-aligner
export HF_TOKEN=hf_xxx            # Hugging Face read token (Svarah is a gated dataset)

# English
python English/preprocess.py --stage all        # select, lexicon, align, features
python English/train.py                         # resumable: run again to continue; --max_minutes N stops cleanly
python English/inference.py --checkpoint English/checkpoints/selected_model_fp16.pth

# Telugu
python Telugu/preprocess.py --stage all
python Telugu/train.py
python Telugu/inference.py --checkpoint Telugu/checkpoints/selected_model_fp16.pth

# score every training snapshot on held-out validation sentences
python common/evaluate_snapshots.py --lang_dir English --ckpt_dir <work>/ckpt --out_dir Results/English/extra/checkpoint_selection
```

Large temporary files go to `<language>/work/` (or `$FS2_WORK`) and are git-ignored. Hyper-parameters, data selection settings and paths are all set in the language's `config.yaml`. `config_v1.yaml` is the configuration of the first run.

Other inference options: `--g2p_mode lexicon` (English: plain dictionary lookup), `--no_normalize` (skip number and time normalisation) and `--all_speakers` (also synthesise with every other speaker).

Training and preprocessing were run in Google Colab on a T4 GPU through the notebooks in `notebooks/`. Inference alone runs on a CPU in a minute or two per language.

## Changes to the reference code (marked `[PATCH]` in the source)

* `audio/stft.py`, `audio/audio_processing.py`: keyword arguments for librosa 0.10 and later, no hard-coded `.cuda()`.
* `text/symbols.py`: 47 Telugu phones added, Mandarin symbols removed.
* `utils/model.py`: vocoder weights unzipped on demand, `map_location`, only HiFi-GAN kept.
* `utils/tools.py`, `model/loss.py`: float32 casts for pitch, energy and mel targets, because the feature files are stored as float64.
* `model/optimizer.py`: AdamW (decoupled weight decay) instead of Adam.
* `preprocessor/preprocessor.py`: explicit train and validation split, statistics from training files only, a fix for a missing alignment file, explicit sampling rate, utf-8.
* The original `train.py`, `synthesize.py`, `evaluate.py` and `prepare_align.py` are replaced by `common/*`: resumable training with CSV loss logs, a snapshot every 1000 steps, variance adaptor plots, and dataset-specific preparation. The model, loss and dataset classes are unchanged.
* Telugu support: `common/telugu_frontend.py` (number, time and ordinal expansion and grapheme-to-phone rules), the Telugu symbols and `Telugu/config.yaml`.

## Training runs

* **Version 1** (`config_v1.yaml`, `Results_v1/`): 30000 steps for English. The Telugu run was stopped at about step 8700 when the GPU quota ran out. The lowest validation loss came very early in both languages (English step 4000, Telugu step 1500), and the audio kept improving after that, so validation loss was not a reliable way to choose a checkpoint.
* **Version 2** (`config.yaml`, `Results/`): 15000 steps, learning rate reduced earlier, AdamW with weight decay 0.01, a snapshot every 1000 steps, and every snapshot scored on held-out validation sentences with `common/evaluate_snapshots.py`. The final checkpoint of each language was chosen after listening to the candidates.

## Notes and limitations

* The assignment PDF prints E3 as "rhyth m" and T3 with a gap inside the word అడవిలో. I synthesised "rhythm" and అడవిలో. The sentences as printed are kept in `test_sentences.json`.
* Svarah is a speech recognition benchmark (conversational speech, many speakers, mixed recording conditions) and not a speech synthesis corpus, so the English model is much weaker than the Telugu model, which was trained on studio recordings.
* The Telugu number words and pronunciation rules were written from standard usage and were not checked by a native speaker.
* The test speaker is the one with the most training data (English `spk013`, Telugu speaker 1).
* The report discusses these limitations and possible improvements in more detail.
