# ComfyUI-HeartMuLa

*Updated Feb 13, 2026: Migrated to V3 Schema & Added Happy New Year 3B Model Support*

A ComfyUI extension for music generation and lyrics transcription based on the [HeartMuLa model family](https://huggingface.co/HeartMuLa) and [heartlib source code](https://github.com/HeartMuLa/heartlib).

## Features
- **V3 Node Schema**: Fully migrated to the modern ComfyUI V3 architecture for improved stability and future-proofing.
- **Latest Model Support**: Integrated the new **HeartMuLa-oss-3B-happy-new-year** model for state-of-the-art music generation.
- **Modular Architecture**: Separate LLM and Codec loaders for better memory management.
- **Inference Optimization**: Integrated `torch.compile` support for Windows, utilizing block-wise compilation to maximize speed without graph breaks. **(Needs correct triton for system)**
- **Text-to-Music**: Generate high-fidelity audio from lyrics and style tags.
- **Lyrics Transcription**: Automatic speech-to-text with support for long-form audio.
- **Folder Picker UI**: Custom folder browser for easy model path selection directly in the UI.

![HeartMuLaGeneration](https://github.com/BobRandomNumber/ComfyUI-HeartMuLa/blob/main/assets/HeartMuLaGeneration.png)

![HeartMuLaTranscription](https://github.com/BobRandomNumber/ComfyUI-HeartMuLa/blob/main/assets/HeartMuLaTranscription.png)

## Installation

1. Navigate to your ComfyUI `custom_nodes` folder:

   ```bash
   cd ComfyUI/custom_nodes
   ```
2. Clone this repository:

   ```bash
   git clone https://github.com/BobRandomNumber/ComfyUI-HeartMuLa.git
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Model Setup

The nodes require specific model weights and configuration files. Create a base folder named `HeartMuLa` inside your ComfyUI models directory (e.g., `ComfyUI/models/HeartMuLa/`) and organize the files as follows.

### 1. Base Configuration Files
Place these files directly in the root of the `HeartMuLa/` folder:
- [gen_config.json](https://huggingface.co/HeartMuLa/HeartMuLaGen/blob/main/gen_config.json)
- [tokenizer.json](https://huggingface.co/HeartMuLa/HeartMuLaGen/blob/main/tokenizer.json)

### 2. Model Directories
Download the repositories below as subfolders inside the `HeartMuLa/` directory.

#### Happy New Year Edition (3B-happy-new-year) - ⭐ RECOMMENDED
- **Model**: [HeartMuLa-oss-3B-happy-new-year](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B-happy-new-year/tree/main)
- **Codec**: [HeartCodec-oss-20260123](https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123/tree/main)

#### For Standard Generation (oss-3B)
- **Model**: [HeartMuLa-oss-3B](https://huggingface.co/HeartMuLa/HeartMuLa-oss-3B/tree/main)
- **Codec**: [HeartCodec-oss](https://huggingface.co/HeartMuLa/HeartCodec-oss/tree/main)

#### For RL-Tuned Generation (RL-oss-3B)
- **Model**: [HeartMuLa-RL-oss-3B-20260123](https://huggingface.co/HeartMuLa/HeartMuLa-RL-oss-3B-20260123/tree/main)
- **Codec**: [HeartCodec-oss-20260123](https://huggingface.co/HeartMuLa/HeartCodec-oss-20260123/tree/main)

#### For Transcription
- **Model**: [HeartTranscriptor-oss](https://huggingface.co/HeartMuLa/HeartTranscriptor-oss/tree/main)

### Final Directory Structure
```text
ComfyUI/models/HeartMuLa/
├── gen_config.json
├── tokenizer.json
├── HeartMuLa-oss-3B-happy-new-year/
├── HeartMuLa-oss-3B/
├── HeartCodec-oss/
├── HeartMuLa-RL-oss-3B-20260123/
├── HeartCodec-oss-20260123/
└── HeartTranscriptor-oss/
```

## Node Descriptions

### 1. HeartMuLa Model Loader
Loads the LLM backbone for music generation.
- **base_path**: Folder containing the model weights. Use the integrated 📁 browser button.
- **model_version**: Select which model version to use.
- **torch_compile**: Enable/Disable `torch.compile` optimization.
- **compile_backend**: Choose the compiler backend (Default: `inductor`).
- **compile_mode**: Choose the optimization level (`default` is best for compatibility).

### 2. HeartMuLa Codec Loader
Loads the audio decoder separately. Runs in standard `fp32` for maximum audio fidelity.
- **base_path**: Folder containing the codec weights. Use the integrated 📁 browser button.
- **codec_version**: Select which codec version to use.

### 3. HeartMuLa Music Generator
The core generation node.
- **lyrics**: The text to be sung or spoken.
- **tags**: Style descriptions (e.g., "piano, happy, wedding, synthesizer, romantic").
- **duration_seconds**: Desired length of the output audio.
- **seed**: Control randomness for reproducible generations.
- **temperature**: Higher values increase creativity/randomness, lower values make it more deterministic.
- **top_k**: Limits sampling to the top K most likely tokens.
- **cfg_scale**: Classifier-Free Guidance scale. Higher values follow tags more strictly (Default: 1.5).

### 4. HeartMuLa Audio Decoder
Converts the generated model tokens into playable audio.

### 5. HeartMuLa Transcription Loader
Loads the Whisper-based lyrics transcription model.
- **base_path**: Folder containing the transcriptor weights. Use the integrated 📁 browser button.

### 6. HeartMuLa Lyrics Transcriber
Converts input audio into text.
- **max_new_tokens**: Maximum length of the generated text.
- **num_beams**: Number of beams for beam search.
- **condition_on_prev_tokens**: If True, uses previous segments as context.
- **logprob_threshold**: Threshold for log probability (Default: -1.0).
- **no_speech_threshold**: Threshold for detecting silent or non-speech segments.
- **temperature**: Sampling temperature (0.0 enables robust multi-temperature decoding).

### 7. Audio Post-Processor
A DSP utility for mastering the generated output.
- **normalize**: Peak normalization to 0dB.
- **stereo_width**: Adjusts stereo image width (Mid-Side processing).
- **high_pass / low_pass**: Removes unwanted frequencies.
- **gain_db**: Adjust output volume.

## Citation

```bibtex
@misc{yang2026heartmulafamilyopensourced,
      title={HeartMuLa: A Family of Open Sourced Music Foundation Models}, 
      author={Dongchao Yang and Yuxin Xie and Yuguo Yin and Zheyu Wang and Xiaoyu Yi and Gongxi Zhu and Xiaolong Weng and Zihan Xiong and Yingzhe Ma and Dading Cong and Jingliang Liu and Zihang Huang and Jinghan Ru and Rongjie Huang and Haoran Wan and Peixu Wang and Kuoxi Yu and Helin Wang and Liming Liang and Xianwei Zhuang and Yuanyuan Wang and Haohan Guo and Junjie Cao and Zeqian Ju and Songxiang Liu and Yuewen Cao and Heming Weng and Yuexian Zou},
      year={2026},
      eprint={2601.10547},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2601.10547}, 
}
```
