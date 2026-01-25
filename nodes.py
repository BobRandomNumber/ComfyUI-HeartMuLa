import os
import torch
import torchaudio
import folder_paths
import comfy.model_management
import comfy.utils
import warnings
import logging
import math

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines.base")
warnings.filterwarnings("ignore", message=".*Mismatch dtype between input and weight.*")
warnings.filterwarnings("ignore", message=".*Key value caches are already setup.*")
warnings.filterwarnings("ignore", message=".*chunk_length_s.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

def resolve_model_path(path):
    if os.path.isabs(path) and os.path.exists(path):
        return path
    h_path = os.path.join(folder_paths.models_dir, "HeartMuLa", path)
    if os.path.exists(h_path): return h_path
    m_path = os.path.join(folder_paths.models_dir, path)
    if os.path.exists(m_path): return m_path
    return path

class HeartMuLaLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_path": ("STRING", {"default": "HeartMuLa", "multiline": False}),
                "model_version": (
                    [
                        "HeartMuLa-RL-oss-3B-20260123",
                        "HeartMuLa-oss-3B",
                    ],
                    {"default": "HeartMuLa-RL-oss-3B-20260123"}
                ),
                "torch_compile": ("BOOLEAN", {"default": False}),
                "compile_backend": (["inductor", "cudagraphs", "eager"], {"default": "inductor"}),
                "compile_mode": (["default", "reduce-overhead", "max-autotune"], {"default": "default"}),
            }
        }

    RETURN_TYPES = ("HEARTMULA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "HeartMuLa"

    def load_model(self, base_path, model_version, torch_compile, compile_backend, compile_mode):
        from .heartlib.pipelines.music_generation import HeartMuLaModel
        resolved_base_path = resolve_model_path(base_path)
        
        if model_version == "HeartMuLa-oss-3B":
            m_folder = "HeartMuLa-oss-3B"
        elif model_version == "HeartMuLa-RL-oss-3B-20260123":
            m_folder = "HeartMuLa-RL-oss-3B-20260123"
        else:
            raise ValueError(f"Unknown model version: {model_version}")

        m_path = os.path.join(resolved_base_path, m_folder)
        if not os.path.exists(m_path):
             raise FileNotFoundError(f"Model folder not found at: {m_path}")

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
        
        print(f"Loading HeartMuLa Generator: {model_version}...")
        try:
            model = HeartMuLaModel.from_pretrained(
                m_path,
                dtype=dtype,
                compile_model=torch_compile,
                compile_backend=compile_backend,
                compile_mode=compile_mode
            )
            return (model,)
        except Exception as e:
            raise RuntimeError(f"Failed to load HeartMuLa model: {e}")

class HeartMuLaCodecLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_path": ("STRING", {"default": "HeartMuLa", "multiline": False}),
                "codec_version": (
                    [
                        "HeartCodec-oss-20260123",
                        "HeartCodec-oss",
                    ],
                    {"default": "HeartCodec-oss-20260123"}
                ),
            }
        }

    RETURN_TYPES = ("HEARTMULA_CODEC",)
    RETURN_NAMES = ("codec",)
    FUNCTION = "load_codec"
    CATEGORY = "HeartMuLa"

    def load_codec(self, base_path, codec_version):
        from .heartlib.pipelines.music_generation import HeartCodecModel
        resolved_base_path = resolve_model_path(base_path)
        c_path = os.path.join(resolved_base_path, codec_version)

        if not os.path.exists(c_path):
             raise FileNotFoundError(f"Codec folder not found at: {c_path}")

        print(f"Loading HeartCodec: {codec_version}...")
        try:
            codec = HeartCodecModel.from_pretrained(c_path)
            return (codec,)
        except Exception as e:
            raise RuntimeError(f"Failed to load HeartCodec: {e}")

class HeartMuLaMusicGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("HEARTMULA_MODEL",),
                "lyrics": ("STRING", {"multiline": True, "default": ""}),
                "tags": ("STRING", {"multiline": True, "default": ""}),
                "duration_seconds": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 300.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0}),
                "top_k": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0}),
            }
        }

    RETURN_TYPES = ("HEARTMULA_TOKENS",)
    RETURN_NAMES = ("tokens",)
    FUNCTION = "generate"
    CATEGORY = "HeartMuLa"

    def generate(self, model, lyrics, tags, duration_seconds, seed, temperature, top_k, cfg_scale):
        torch.manual_seed(seed)
        try:
            torch.set_float32_matmul_precision('high')
            torch._dynamo.reset()
        except: pass
            
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        device = comfy.model_management.get_torch_device()
        
        cleaned_tags = ",".join([t.strip() for t in tags.split(",") if t.strip()])
        
        try:
            inputs = {"lyrics": lyrics, "tags": cleaned_tags}
            processed_inputs = model.preprocess(inputs, cfg_scale)
            max_audio_length_ms = int(duration_seconds * 1000)

            print(f"Moving HeartMuLa Generator to {device}...")
            model.model.to(device)
            model.ensure_compiled()
            
            for k, v in processed_inputs.items():
                if isinstance(v, torch.Tensor):
                    processed_inputs[k] = v.to(device)
                    if v.is_floating_point():
                         processed_inputs[k] = processed_inputs[k].to(dtype=model.dtype)

            batch_size = processed_inputs["tokens"].shape[0]
            # Use the new safe wrapper that avoids redundant setup warnings
            model.setup_caches(max_batch_size=batch_size)

            gen_steps = int(duration_seconds * 1000 // 80)
            pbar = comfy.utils.ProgressBar(gen_steps)
            
            def gen_callback(step):
                pbar.update(1)

            with torch.no_grad():
                tokens = model.generate_tokens(
                    processed_inputs,
                    max_audio_length_ms=max_audio_length_ms,
                    temperature=temperature,
                    topk=top_k,
                    cfg_scale=cfg_scale,
                    callback=gen_callback
                )
            
            torch.cuda.synchronize()
            return (tokens,)
        finally:
            model.model.to("cpu")
            torch.cuda.synchronize()
            comfy.model_management.soft_empty_cache()

class HeartMuLaAudioDecoder:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tokens": ("HEARTMULA_TOKENS",),
                "codec": ("HEARTMULA_CODEC",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "decode"
    CATEGORY = "HeartMuLa"

    def decode(self, tokens, codec):
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        device = comfy.model_management.get_torch_device()
        
        try:
            if tokens is None or tokens.numel() == 0:
                print("Warning: No tokens generated. Skipping decode.")
                return ({"waveform": torch.zeros(1, 1, 48000), "sample_rate": 48000},)

            print(f"Moving HeartCodec to {device}...")
            codec.audio_codec.to(device)
            
            # Ensure tokens are on the correct device
            tokens = tokens.to(device)

            # Robustness: The codec expects indices in range [0, 8191].
            # Sometimes the LLM predicts an EOS token (8193) or other high IDs.
            # We clip these to the maximum valid codebook index to prevent out-of-bounds crashes.
            tokens = torch.clamp(tokens, 0, 8191)

            # ProgressBar for decoding
            codes_len = tokens.shape[-1]
            actual_decode_chunks = (codes_len - 104 + 319) // 320 # Approx
            pbar = comfy.utils.ProgressBar(max(1, actual_decode_chunks * 10))
            
            def decode_callback(step):
                pbar.update(1)

            with torch.no_grad():
                outputs = codec.decode_tokens(tokens, duration=29.76, callback=decode_callback)
            
            torch.cuda.synchronize()
            wav = outputs["wav"]
            if wav.dim() == 1: wav = wav.unsqueeze(0)
            wav = wav.unsqueeze(0) # ComfyUI AUDIO format: [B, C, T]
            
            return ({"waveform": wav.cpu(), "sample_rate": 48000},)
        finally:
            codec.audio_codec.to("cpu")
            torch.cuda.synchronize()
            comfy.model_management.soft_empty_cache()

class HeartMuLaTranscriptionLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_path": ("STRING", {"default": "HeartMuLa", "multiline": False}),
            }
        }

    RETURN_TYPES = ("HEARTMULA_TRANSCRIPTOR",)
    RETURN_NAMES = ("transcriptor",)
    FUNCTION = "load_model"
    CATEGORY = "HeartMuLa"

    def load_model(self, base_path):
        from .heartlib.pipelines.lyrics_transcription import HeartTranscriptorPipeline
        target_path = resolve_model_path(base_path)
        if not os.path.exists(target_path):
             raise FileNotFoundError(f"Model path does not exist: {target_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Loading HeartMuLa Transcriptor from {target_path}...")
        try:
            pipeline = HeartTranscriptorPipeline.from_pretrained(target_path, device=device, dtype=dtype)
            return (pipeline,)
        except Exception as e:
            raise RuntimeError(f"Failed to load HeartMuLa Transcriptor: {e}")

class HeartMuLaLyricsTranscriber:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "transcriptor": ("HEARTMULA_TRANSCRIPTOR",),
                "audio": ("AUDIO",),
                "max_new_tokens": ("INT", {"default": 256, "min": 1, "max": 445, "step": 1}),
                "num_beams": ("INT", {"default": 2, "min": 1, "max": 5, "step": 1}),
                "condition_on_prev_tokens": ("BOOLEAN", {"default": False}),
                "logprob_threshold": ("FLOAT", {"default": -1.0, "min": -20.0, "max": 0.0, "step": 0.1}),
                "no_speech_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "transcribe"
    CATEGORY = "HeartMuLa"

    def transcribe(self, transcriptor, audio, max_new_tokens, num_beams, condition_on_prev_tokens, logprob_threshold, no_speech_threshold, temperature):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        wav = waveform.cpu()
        if wav.shape[1] > 1: wav = wav.mean(dim=1, keepdim=True)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            wav = resampler(wav)
            sample_rate = 16000
        wav = wav.squeeze(0)
        wav_np = wav.numpy()
        if temperature == 0.0:
            temperature_arg = (0.0, 0.1, 0.2, 0.4)
        else:
            temperature_arg = temperature
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "task": "transcribe",
            "condition_on_prev_tokens": condition_on_prev_tokens,
            "compression_ratio_threshold": 1.8,
            "temperature": temperature_arg,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
        }
        print(f"Moving Whisper to {transcriptor.device}...")
        transcriptor.model.to(transcriptor.device)
        try:
            result = transcriptor({"raw": wav_np, "sampling_rate": sample_rate}, generate_kwargs=generate_kwargs)
            return (result["text"],)
        finally:
            print("Moving Whisper to CPU...")
            transcriptor.model.to("cpu")
            comfy.model_management.soft_empty_cache()

class HeartMuLaPostProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "normalize": ("BOOLEAN", {"default": True}),
                "stereo_width": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "high_pass": ("INT", {"default": 80, "min": 0, "max": 1000, "step": 1}),
                "low_pass": ("INT", {"default": 18000, "min": 0, "max": 24000, "step": 100}),
                "gain_db": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "process"
    CATEGORY = "HeartMuLa"

    def process(self, audio, normalize, stereo_width, high_pass, low_pass, gain_db):
        waveform = audio["waveform"].clone()
        sample_rate = audio["sample_rate"]
        device = waveform.device
        waveform = waveform.cpu()
        if high_pass > 0: waveform = torchaudio.functional.highpass_biquad(waveform, sample_rate, high_pass)
        if low_pass > 0 and low_pass < (sample_rate // 2): waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate, low_pass)
        if gain_db != 0:
            ratio = 10 ** (gain_db / 20)
            waveform = waveform * ratio
        if stereo_width != 1.0:
            if waveform.shape[1] == 1: waveform = waveform.repeat(1, 2, 1)
            mid = (waveform[:, 0:1, :] + waveform[:, 1:2, :]) / 2.0
            side = (waveform[:, 0:1, :] - waveform[:, 1:2, :]) / 2.0
            side = side * stereo_width
            new_l = mid + side
            new_r = mid - side
            waveform = torch.cat([new_l, new_r], dim=1)
        if normalize:
            max_val = torch.abs(waveform).max()
            if max_val > 1e-6: waveform = waveform / max_val
        return ({"waveform": waveform.to(device), "sample_rate": sample_rate},)

NODE_CLASS_MAPPINGS = {
    "HeartMuLaLoader": HeartMuLaLoader,
    "HeartMuLaCodecLoader": HeartMuLaCodecLoader,
    "HeartMuLaMusicGenerator": HeartMuLaMusicGenerator,
    "HeartMuLaAudioDecoder": HeartMuLaAudioDecoder,
    "HeartMuLaTranscriptionLoader": HeartMuLaTranscriptionLoader,
    "HeartMuLaLyricsTranscriber": HeartMuLaLyricsTranscriber,
    "HeartMuLaPostProcessor": HeartMuLaPostProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HeartMuLaLoader": "HeartMuLa Loader",
    "HeartMuLaCodecLoader": "HeartMuLa Codec Loader",
    "HeartMuLaMusicGenerator": "HeartMuLa Music Generator",
    "HeartMuLaAudioDecoder": "HeartMuLa Audio Decoder",
    "HeartMuLaTranscriptionLoader": "HeartMuLa Transcription Loader",
    "HeartMuLaLyricsTranscriber": "HeartMuLa Lyrics Transcriber",
    "HeartMuLaPostProcessor": "Audio Post-Processor"
}