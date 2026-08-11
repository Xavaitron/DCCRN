# Audio Visual Zooming on Low compute devices using Deep Complex Convolutional Recurrent Network (DCCRN)

Audio-visual (AV) zooming aligns the auditory field
of view with that of the camera by enhancing sources lying
inside the visual frustum and suppressing those outside. On
consumer smartphones, only two closely spaced microphones
are available, computational budgets are limited, and reverber-
ation distorts the spatial cues on which classical beamforming
relies. We present a complete mathematical and algorithmic
specification of a lightweight (∼ 10M parameters) direction-
conditioned enhancement network, the DCCRN–Conformer, that
addresses these constraints. The system operates in the complex
short-time Fourier transform (STFT) domain. A complex-valued
encoder–decoder retains the inter-channel phase difference (IPD)
carrying direction-of-arrival information, a dual-path Conformer
bottleneck performs joint frequency- and time-axis modeling,
and a unit-circle azimuth embedding controls a shared real-
valued multiplicative bottleneck gain. On 5,000 reverberant test
mixtures (RT60 = 0.5 s, SIR = 0 dB, SNR = 5 dB), the proposed
system improves SI-SDR by 40.14 dB, STOI by 0.214, and PESQ
by 1.28 over the unprocessed mixture, while processing a three-
second segment in approximately 50.55 ms on the reported CPU
setup.

---

## 📁 Project Structure

```
├── Dataset Generation/              # MATLAB scripts for synthetic dataset creation
│   ├── train_anechoic.m             # Training data (150k samples, RT60=0.0)
│   ├── train_reverb.m               # Training data (150k samples, RT60=0.5)
│   ├── test_anechoic.m              # Test data (5k samples, fixed 90°/40° angles)
│   └── test_reverb.m                # Test data (5k samples, fixed 90°/40° angles)
│
├── Model Inference/                 # Python training, testing, and inference
│   ├── train_Conformer.py           # Training script
│   ├── test_Conformer.py            # Evaluation script (SI-SDR, STOI, PESQ)
│   ├── inference_Conformer.py       # Single-file inference
│   ├── anechoic_Conformer.pth       # Trained model (anechoic)
│   ├── reverb_Conformer.pth         # Trained model (reverberant)
│   ├── evaluation_anechoic/         # Evaluation outputs
│   └── evaluation_reverb/           # Evaluation outputs
│
├── RIR_Gen/                         # Room Impulse Response generator (MEX)
├── generate_rir_data.m              # Generates rir_data.mat for inference
├── prepare_submission.m             # Packages evaluation outputs
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## 🔧 Requirements

### Python
```bash
pip install -r requirements.txt
```

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.6.0 | Deep learning |
| torchaudio | 2.6.0 | Audio I/O |
| torchmetrics | 1.8.2 | PESQ, STOI, SI-SDR |
| soundfile | latest | Audio backend |
| pesq, pystoi | latest | Metrics |

### MATLAB
- MATLAB R2020b+
- Signal Processing Toolbox
- Parallel Computing Toolbox
- `rir_generator` MEX function

### RIR Generator Setup

The `rir_generator` MEX function needs to be compiled before running dataset generation:

```matlab
% 1. Navigate to RIR_gen folder
cd RIR_gen

% 2. Configure MEX compiler for C++
mex -setup

% Select a C++ compiler (MinGW-w64, MSVC, etc. must be installed)

% 3. Compile the RIR generator
mex rir_generator.cpp rir_generator_core.cpp
```

> **Note:** On Windows, install [MinGW-w64](https://www.mingw-w64.org/) or Visual Studio with C++ build tools. On Linux/macOS, ensure `g++` or `clang++` is available.

---

## 🚀 Pipeline

### 1. Download Raw Dataset

Download the raw audio files needed for dataset generation:

1. Download `Dataset_raw.zip` from [Google Drive](https://drive.google.com/file/d/1hG6gk2BDHD-96WnAUOxVxm_p86jbgX9r/view?usp=sharing)
2. Place the zip file in the project root
3. Extract it so the folder structure looks like:

```
Dataset_raw/
├── Male/       # Male speech files (.wav/.flac)
├── Female/     # Female speech files (.wav/.flac)
├── Noise/      # Noise files (.wav/.flac)
└── Music/      # Music files (.wav/.flac)
```

> **Note:** `Dataset_raw/` and `Dataset_raw.zip` are gitignored and will not be committed.

### 2. Dataset Generation (MATLAB)

```matlab
cd "Dataset Generation"
train_anechoic   % 150k samples, RT60=0.0, random angles
train_reverb     % 150k samples, RT60=0.5, random angles
test_anechoic    % 5k samples, RT60=0.0, fixed angles (90°/40°)
test_reverb      % 5k samples, RT60=0.5, fixed angles (90°/40°)
```

**Output per sample:**
```
sample_XXXXX/
├── mixture.wav      # Stereo (target + interferer + noise)
├── target.wav       # Ground truth
├── interference.wav # Interference
└── meta.json        # {target_angle, interf_angle, rt60, ...}
```

**Settings:** SIR = 0 dB, SNR = 5 dB, 16 kHz, 4 s duration

---

### 3. Training

```bash
cd "Model Inference"
python train_Conformer.py
```

Edit config in script:
```python
DATASET_ROOT = r"../Train_Dataset/reverb"  # or anechoic
RESUME_FROM = "reverb_Conformer.pth"       # or None
```

**Training details:**
- Optimizer: AdamW (lr = 1e-4, weight decay = 1e-4)
- Scheduler: Cosine Annealing (η_min = 1e-6)
- Loss: Composite SI-SDR + Multi-Resolution STFT + Mel-Spectrogram + Phase-Aware
- Silence augmentation: 20% probability (random off-axis angles)
- Epochs: 50, Batch size: 4

---

### 4. Evaluation

```bash
cd "Model Inference"
python test_Conformer.py
```

Edit config:
```python
MODEL_PATH = "anechoic_Conformer.pth"
TEST_DATASET_ROOT = r"../Test_Dataset/anechoic"
OUTPUT_DIR = "evaluation_anechoic"
```

**Outputs:** Best samples by category (Overall, Male+Female, Male+Music, Male+Noise)

---

### 5. Single-File Inference

```bash
python inference_Conformer.py -i input.wav -a 90 -o output.wav -m reverb_Conformer.pth -d cuda
```

| Arg | Description |
|-----|-------------|
| `-i` | Input stereo audio |
| `-a` | Target angle (0–180°) |
| `-o` | Output file |
| `-m` | Model checkpoint |
| `-d` | Device (cpu/cuda) |

---

### 6. Generate RIR Data

```bash
matlab -batch "run('generate_rir_data.m')"
```

Generates `rir_data.mat` containing Room Impulse Responses for:
- **Anechoic** (RT60 = 0.0)
- **Reverberant** (RT60 = 0.5)

---

## 🏗️ Model Architecture

**DCCRN-Conformer** — 9,994,312 parameters (~10M)

```
Stereo Input (2ch, 16kHz)
    │
    ▼
  STFT (n_fft=512, hop=128)
    │
    ▼
┌─────────────────────────────────┐
│  Complex Encoder                │
│  Conv2d: 2 → 48 → 96 → 192 → 256  │
│  + ComplexBatchNorm + SE Blocks │
└─────────────────────────────────┘
    │
    ├── Angle MLP Injection (sin/cos → 128 → 256 → 256)
    │
    ▼
┌─────────────────────────────────┐
│  Dual-Path Conformer Bottleneck │
│  3 blocks × 4 heads             │
│  Freq-path (k=15) + Time-path (k=31) │
│  FFN → MHSA → ConvModule → FFN │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Complex Decoder                │
│  ConvTranspose2d + Skip Connections │
│  256 → 192 → 96 → 48 → 2      │
└─────────────────────────────────┘
    │
    ▼
  Complex Masking (tanh)
    │
    ▼
  iSTFT → Mono Output
```

| Component | Details |
|-----------|---------|
| Encoder | Complex Conv2d: 2→48→96→192→256 with Squeeze-Excitation |
| Bottleneck | Dual-Path Conformer (3 blocks, 4 heads, depthwise separable conv) |
| Decoder | Complex ConvTranspose2d with skip connections |
| Conditioning | Angle MLP: sin/cos encoding → 128 → 256 → 256, injected at bottleneck |
| Masking | Complex ratio masking with tanh activation |
| Audio | 16 kHz, STFT n_fft=512, hop=128, 3 s fixed input |

---

## 📊 Evaluation Results

All results are reported on 5,000 test samples per condition with fixed geometry (target at 90°, interferer at 40°), SIR = 0 dB, SNR = 5 dB.

### Anechoic Condition (RT60 = 0.0)

| Metric | Average | Best Case |
|--------|---------|-----------|
| **SI-SDR** | 12.46 dB | 16.19 dB |
| **STOI** | 0.8812 | 0.9760 |
| **PESQ** | 1.6131 | 2.8012 |

- **Best Overall Sample:** test_sample_00504 (Combined Score: 2.92 / 3.0)
- **Inference:** 50.64 ms avg ± 5.58 ms → **59.24× real-time** (for 3 s audio)

### Reverberant Condition (RT60 = 0.5)

| Metric | Average | Best Case |
|--------|---------|-----------|
| **SI-SDR** | 8.91 dB | 11.19 dB |
| **STOI** | 0.8419 | 0.9627 |
| **PESQ** | 1.4456 | 2.2263 |

- **Best Overall Sample:** test_sample_00404 (Combined Score: 2.81 / 3.0)
- **Inference:** 50.66 ms avg ± 5.40 ms → **59.22× real-time** (for 3 s audio)

### Summary

| Condition | SI-SDR (dB) | STOI | PESQ | Real-time Factor |
|-----------|-------------|------|------|------------------|
| Anechoic (RT60 = 0.0) | 12.46 | 0.8812 | 1.6131 | 59.24× |
| Reverberant (RT60 = 0.5) | 8.91 | 0.8419 | 1.4456 | 59.22× |

---

## 📏 Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **SI-SDR** | Scale-Invariant Signal-to-Distortion Ratio | dB (higher is better) |
| **STOI** | Short-Time Objective Intelligibility | 0–1 (higher is better) |
| **PESQ** | Perceptual Evaluation of Speech Quality | −0.5 to 4.5 (higher is better) |

---

## 📋 Quick Start

```bash
# Setup
pip install -r requirements.txt

# Single-file inference
cd "Model Inference"
python inference_Conformer.py -i audio.wav -a 90 -o out.wav -d cuda

# Full evaluation
python test_Conformer.py
```

---

## 📄 License

This project is intended for research purposes. If you use this work, please cite accordingly.
