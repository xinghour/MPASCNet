# MPASCNet: Magnitude-Phase-Aware Adaptive Spectral Compression for Compressed Music Restoration

PyTorch implementation of **MPASCNet**, a magnitude-phase-aware adaptive spectral compression network for compressed music restoration.

<p align="center">
  <img src="assets/mpascnet_architecture.svg" width="100%">
</p>

## Results

### Restoration Performance on MedleyDB

| Method | SDR ↑ | SI-SNR ↑ | ViSQOL ↑ | Params (M) | RTF (×10⁻³) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MP3 Input | 13.92 | 12.22 | 3.02 | - | - |
| AEROMambaPS | 16.26 | 15.73 | 3.49 | 18.55 | **4.10** |
| Apollo | 16.61 | 15.77 | 3.60 | 16.54 | 37.54 |
| **MPASCNet** | **17.65** | **17.27** | **3.67** | **6.49** | 23.06 |

### Spectral Reconstruction

| Method | PD ↓ | LSD ↓ |
| --- | ---: | ---: |
| AEROMambaPS | 16.62 | 1.12 |
| Apollo | 14.88 | 1.04 |
| **MPASCNet** | **13.80** | **0.92** |

## Preparation

### Environment

We recommend using Python 3.10.

```bash
conda create -n mpascnet python=3.10
conda activate mpascnet

pip install -r requirements.txt
```

A working MP3 codec environment is required for MP3 data generation.

ViSQOL is an optional dependency and is not included in `requirements.txt`. Please install ViSQOL separately if ViSQOL evaluation is required.

### Pretrained Checkpoint

The pretrained MPASCNet checkpoint is available on Hugging Face:

https://huggingface.co/Xinghour/MPASCNet

The default inference configuration expects the checkpoint to be placed under:

```text
cp_model/
└── best
```

## Inference

### Single Audio

MPASCNet can restore a single compressed audio file:

```bash
python inference.py \
    --input path/to/input.mp3 \
    --checkpoint_path cp_model
```

The restored audio will be saved to `inference_results/` by default.

### Audio Folder

A directory containing multiple audio files can also be processed:

```bash
python inference.py \
    --input path/to/audio_folder \
    --checkpoint_path cp_model
```

## Training

### Data Preparation

MPASCNet is trained using **MUSDB18-HQ** and **MoisesDB** following the data preparation protocol of Apollo.

The preprocessing scripts are located in:

```text
data/create_train/
├── merge.py
├── moisesdb_preprocess.py
└── musdb_preprocess.py
```

The MUSDB18-HQ and MoisesDB preprocessing scripts are based on the data preparation code released with Apollo.

#### 1. Preprocess MoisesDB

Set the dataset and output paths in:

```text
data/create_train/moisesdb_preprocess.py
```

and run:

```bash
python data/create_train/moisesdb_preprocess.py
```

The processed HDF5 files are organized by instrument category.

#### 2. Preprocess MUSDB18-HQ

Set the MUSDB18-HQ train, test, and output paths in:

```text
data/create_train/musdb_preprocess.py
```

and run:

```bash
python data/create_train/musdb_preprocess.py
```

The MUSDB18-HQ data contain the following categories:

```text
bass
drums
other
vocals
mixture
```

#### 3. Merge MUSDB18-HQ and MoisesDB

MoisesDB is used as the base dataset.

For the four overlapping categories:

```text
bass
drums
other
vocals
```

the MUSDB18-HQ HDF5 files are appended to the corresponding MoisesDB directories with continuous file indices.

For example, if the last file in the MoisesDB `bass` directory is:

```text
sample198.hdf5
```

the MUSDB18-HQ `bass` files will be copied as:

```text
sample199.hdf5
sample200.hdf5
...
```

The index is determined independently for each category.

Since the processed MoisesDB data do not contain a `mixture` category, the MUSDB18-HQ mixture files are copied directly to the final training directory.

Run:

```bash
python data/create_train/merge.py \
    --moises_dir path/to/moises_hdf5 \
    --musdb_dir path/to/musdb18hq_hdf5 \
    --output_dir path/to/training_data
```

The resulting training directory follows the structure:

```text
training_data/
├── bass/
├── bowed_strings/
├── drums/
├── guitar/
├── other/
├── other_keys/
├── other_plucked/
├── percussion/
├── piano/
├── vocals/
├── wind/
└── mixture/
```

During training, active stems are randomly sampled and mixed, and MP3 compression is applied with bitrates selected from:

```text
24, 32, 48, 64, 96, 128 kbps
```

### Training

Start training with:

```bash
python train.py \
    --train_dir path/to/training_data \
    --eval_dir path/to/validation_data \
    --checkpoint_path cp_model
```

The main training configuration is defined in:

```text
config.json
```

W&B logging is set to offline mode by default.

For multi-GPU training, specify the GPU IDs in `config.json`.

The best checkpoint and the latest training checkpoint are saved as:

```text
cp_model/
├── best
└── latest
```

Training uses early stopping when the validation SI-SNR does not improve for 20 consecutive epochs.

## Evaluation

### Evaluation Data

The evaluation set is constructed from **MedleyDB**.

To avoid overlap with the training data, the 46 tracks shared between MedleyDB and MUSDB18-HQ are excluded, leaving 150 tracks for evaluation.

We evaluate three conditions:

- **Multi-Stem**: original mixtures
- **Single-Stem**: individual processed stems
- **Vocal**: vocal stems

For each condition, 500 three-second excerpts are deterministically sampled and degraded using MP3 compression.

The evaluation data preparation scripts are located in:

```text
data/
├── create_test.py
└── create_val.py
```

### SDR and SI-SNR

For evaluation data organized in the format expected by `MusicCodecEvalDataset`, run:

```bash
python inference.py \
    --eval_dir path/to/evaluation_data \
    --checkpoint_path cp_model \
    --output_dir inference_results
```

The script reports SDR and SI-SNR and saves the restored audio and evaluation results.

### PD and LSD

Spectral reconstruction can be evaluated using:

```bash
python losses/pdlsd.py \
    --eval_dir path/to/evaluation_data \
    --enhanced_dir path/to/inference_results \
    --output_path pd_lsd.csv
```

### ViSQOL

ViSQOL is not included in `requirements.txt` and requires a separate installation.

After installing ViSQOL, the corresponding evaluation can be performed using:

```bash
python losses/visqol.py \
    --eval_dir path/to/evaluation_data \
    --enhanced_dir path/to/inference_results \
    --output_dir visqol_results
```

## Acknowledgements

The training data preparation pipeline is based on the preprocessing code released with [Apollo](https://github.com/jusperlee/apollo).

MPASCNet is also inspired by prior work on magnitude-phase modeling and adaptive spectral feature compression.

We thank the authors and contributors of the following projects and datasets:

- Apollo
- MP-SENet
- Spectral Feature Compression
- MUSDB18-HQ
- MoisesDB
- MedleyDB

## License

The license for this repository will be provided with the public release.
