- # One-Click-MB-iSTFT-VITS2

This tool allows you to complete the entire process of MB-iSTFT-VITS2 (Data Preprocessing + Whisper ASR + Text Preprocessing + Modification config.json + Training, Inference) with ONE-CLICK!



## Table of Contents 
- [Structure](#structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Prepare_Datasets](#prepare_datasets)
- [Usage](#usage)
- [Inference](#inference)
- [Changes](#changes)
- [References](#references)

## Structure
![Sample Output](./src/image5.png)

## Prerequisites
- A Windows/Linux system with a minimum of `16GB` RAM.
- A GPU with at least `12GB` of VRAM.
- Python == 3.8
- Anaconda installed.
- PyTorch installed.
- CUDA 11.x installed.
- Zlib DLL installed.

Pytorch install command:
```sh
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

CUDA 11.7 Install:
`https://developer.nvidia.com/cuda-11-7-0-download-archive`

Zlib DLL Install:
`https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-zlib-windows`

Install pyopenjtalk Manually:
`pip install -U pyopenjtalk --no-build-isolation`

---

## Installation 
1. **Create an Anaconda environment:**

```sh
conda create -n vits2 python=3.8
```

2. **Activate the environment:**

```sh
conda activate vits2
```

3. **Clone this repository to your local machine:**

```sh
git clone https://github.com/ORI-Muchim/One-Click-MB-iSTFT-VITS2.git
```

4. **Navigate to the cloned directory:**

```sh
cd One-Click-MB-iSTFT-VITS2
```

5. **Install the necessary dependencies:**

```sh
pip install -r requirements.txt
```

---

## Prepare_Datasets

Place the audio files as follows. 

.mp3 or .wav files are okay.

```
One-Click-MB-iSTFT-VITS2
├────datasets
│       ├───speaker0
│       │   ├────1.mp3
│       │   └────1.wav
│       └───speaker1
│       │    ├───1.mp3
│       │    └───1.wav
│       ├integral.py
│       └integral_low.py
│
├────vits2
├────inference.py
├────main_low.py
├────main_resume.py
├────main.py
├────Readme.md
└────requirements.txt
```

This is just an example, and it's okay to add more speakers.

---

## Usage

To start this tool, use the following command, replacing {language}, {model_name}, and {sample_rate} with your respective values({language: ko, ja, en, zh} / {sample_rate: 22050 / 44100}):

```sh
python main.py {language} {model_name} {sample_rate}
```

For those with low specifications(VRAM <= 16GB), please use this code:

```sh
python main_low.py {language} {model_name} {sample_rate}
```

If the data configuration is complete and you want to resume training, enter this code:

```sh
python main_resume.py {model_name}
```

---
## Inference

After the model has been trained, you can generate predictions by using the following command, replacing {model_name} and {model_step} with your respective values:

```sh
python inference.py {model_name} {model_step} --poly(Optional)
```

If you have selected cjke_cleaners2(en, zh), put `--poly` option at the back.


If you want to change the example text used in the reference, modify `./vits/inference.py` `input` part.

---

## Changes

In the repository of [CjangCjengh/vits](https://github.com/CjangCjengh/vits.git), I made some modifications to the Korean text cleaning method. The other cleaning process is the same by posting it to the CjangCjengh repository, but the cleaner file was modified using the [tenebo/g2pk2](https://github.com/tenebo/g2pk2) library as Korean pronounced.

---
## References

For more information, please refer to the following repositories: 
- [jaywalnut310/vits](https://github.com/jaywalnut310/vits.git) 
- [CjangCjengh/vits](https://github.com/CjangCjengh/vits.git)
- [FENRlR/MB-iSTFT-VITS2](https://github.com/FENRlR/MB-iSTFT-VITS2)
- [tenebo/g2pk2](https://github.com/tenebo/g2pk2)
- [henrymass/AudioSlicer](https://github.com/henrymaas/AudioSlicer)
