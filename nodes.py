import os
import torch
import torchaudio
import folder_paths
import comfy.model_management
import comfy.utils
import warnings
import logging
import math
from comfy_api.latest import io, ui

warnings.filterwarnings("ignore", category=UserWarning, module="transformers.pipelines.base")
warnings.filterwarnings("ignore", message=".*Mismatch dtype between input and weight.*")
warnings.filterwarnings("ignore", message=".*Key value caches are already setup.*")
warnings.filterwarnings("ignore", message=".*chunk_length_s.*")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Custom Types for V3
HEARTMULA_MODEL = io.Custom("HEARTMULA_MODEL")
HEARTMULA_CODEC = io.Custom("HEARTMULA_CODEC")
HEARTMULA_TOKENS = io.Custom("HEARTMULA_TOKENS")
HEARTMULA_TRANSCRIPTOR = io.Custom("HEARTMULA_TRANSCRIPTOR")
AUDIO = io.Custom("AUDIO")

def resolve_model_path(path):
    if os.path.isabs(path) and os.path.exists(path):
        return path
    h_path = os.path.join(folder_paths.models_dir, "HeartMuLa", path)
    if os.path.exists(h_path): return h_path
    m_path = os.path.join(folder_paths.models_dir, path)
    if os.path.exists(m_path): return m_path
    return path

class HeartMuLaLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HeartMuLaLoader",
            display_name="HeartMuLa Loader",
            category="HeartMuLa",
            inputs=[
                io.String.Input("base_path", default="HeartMuLa", multiline=False),
                io.Combo.Input("model_version", options=[
                    "HeartMuLa-oss-3B-happy-new-year",
                    "HeartMuLa-RL-oss-3B-20260123",
                    "HeartMuLa-oss-3B",
                ], default="HeartMuLa-oss-3B-happy-new-year"),
                io.Boolean.Input("torch_compile", default=False),
                io.Combo.Input("compile_backend", options=["inductor", "cudagraphs", "eager"], default="inductor"),
                io.Combo.Input("compile_mode", options=["default", "reduce-overhead", "max-autotune"], default="default"),
            ],
            outputs=[HEARTMULA_MODEL.Output(display_name="model")]
        )

    @classmethod
    def execute(cls, base_path, model_version, torch_compile, compile_backend, compile_mode) -> io.NodeOutput:
        from .heartlib.pipelines.music_generation import HeartMuLaModel
        resolved_base_path = resolve_model_path(base_path)
        m_path = os.path.join(resolved_base_path, model_version)
            
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
            return io.NodeOutput(model)
        except Exception as e:
            raise RuntimeError(f"Failed to load HeartMuLa model: {e}")

class HeartMuLaCodecLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HeartMuLaCodecLoader",
            display_name="HeartMuLa Codec Loader",
            category="HeartMuLa",
            inputs=[
                io.String.Input("base_path", default="HeartMuLa", multiline=False),
                io.Combo.Input("codec_version", options=[
                    "HeartCodec-oss-20260123",
                    "HeartCodec-oss",
                ], default="HeartCodec-oss-20260123"),
            ],
            outputs=[HEARTMULA_CODEC.Output(display_name="codec")]
        )

    @classmethod
    def execute(cls, base_path, codec_version) -> io.NodeOutput:
        from .heartlib.pipelines.music_generation import HeartCodecModel
        resolved_base_path = resolve_model_path(base_path)
        c_path = os.path.join(resolved_base_path, codec_version)

        if not os.path.exists(c_path):
             raise FileNotFoundError(f"Codec folder not found at: {c_path}")

        print(f"Loading HeartCodec: {codec_version}...")
        try:
            codec = HeartCodecModel.from_pretrained(c_path)
            return io.NodeOutput(codec)
        except Exception as e:
            raise RuntimeError(f"Failed to load HeartCodec: {e}")

class HeartMuLaMusicGenerator(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HeartMuLaMusicGenerator",
            display_name="HeartMuLa Music Generator",
            category="HeartMuLa",
            inputs=[
                HEARTMULA_MODEL.Input("model"),
                io.String.Input("lyrics", multiline=True, default=""),
                io.String.Input("tags", multiline=True, default=""),
                io.Float.Input("duration_seconds", default=30.0, min=1.0, max=300.0),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff),
                io.Float.Input("temperature", default=1.0, min=0.1, max=2.0),
                io.Int.Input("top_k", default=50, min=1, max=1000),
                io.Float.Input("cfg_scale", default=1.5, min=0.1, max=5.0),
            ],
            outputs=[HEARTMULA_TOKENS.Output(display_name="tokens")]
        )

    @classmethod
    def execute(cls, model, lyrics, tags, duration_seconds, seed, temperature, top_k, cfg_scale) -> io.NodeOutput:
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
            torch.cuda.synchronize()
            
            model.setup_caches(max_batch_size=processed_inputs["tokens"].shape[0])
            model.ensure_compiled()

            for k, v in processed_inputs.items():
                if isinstance(v, torch.Tensor):
                    processed_inputs[k] = v.to(device)
                    if v.is_floating_point():
                         processed_inputs[k] = processed_inputs[k].to(dtype=model.dtype)

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
            return io.NodeOutput(tokens)
        finally:
            model.model.to("cpu")
            torch.cuda.synchronize()
            comfy.model_management.soft_empty_cache()

class HeartMuLaAudioDecoder(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HeartMuLaAudioDecoder",
            display_name="HeartMuLa Audio Decoder",
            category="HeartMuLa",
            inputs=[
                HEARTMULA_TOKENS.Input("tokens"),
                HEARTMULA_CODEC.Input("codec"),
            ],
            outputs=[AUDIO.Output()]
        )

    @classmethod
    def execute(cls, tokens, codec) -> io.NodeOutput:
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        device = comfy.model_management.get_torch_device()
        
        try:
            if tokens is None or tokens.numel() == 0:
                print("Warning: No tokens generated. Skipping decode.")
                return io.NodeOutput({"waveform": torch.zeros(1, 1, 48000), "sample_rate": 48000})

            print(f"Moving HeartCodec to {device}...")
            codec.audio_codec.to(device)
            tokens = tokens.to(device)
            tokens = torch.clamp(tokens, 0, 8191)

            codes_len = tokens.shape[-1]
            actual_decode_chunks = (codes_len - 104 + 319) // 320
            pbar = comfy.utils.ProgressBar(max(1, actual_decode_chunks * 10))
            
            def decode_callback(step):
                pbar.update(1)

            with torch.no_grad():
                outputs = codec.decode_tokens(tokens, duration=29.76, callback=decode_callback)
            
            torch.cuda.synchronize()
            wav = outputs["wav"]
            if wav.dim() == 1: wav = wav.unsqueeze(0)
            wav = wav.unsqueeze(0)
            
            return io.NodeOutput({"waveform": wav.cpu(), "sample_rate": 48000})
        finally:
            codec.audio_codec.to("cpu")
            torch.cuda.synchronize()
            comfy.model_management.soft_empty_cache()

class HeartMuLaTranscriptionLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HeartMuLaTranscriptionLoader",
            display_name="HeartMuLa Transcription Loader",
            category="HeartMuLa",
            inputs=[
                io.String.Input("base_path", default="HeartMuLa", multiline=False),
            ],
            outputs=[HEARTMULA_TRANSCRIPTOR.Output(display_name="transcriptor")]
        )

    @classmethod
    def execute(cls, base_path) -> io.NodeOutput:
        from .heartlib.pipelines.lyrics_transcription import HeartTranscriptorPipeline
        target_path = resolve_model_path(base_path)
        if not os.path.exists(target_path):
             raise FileNotFoundError(f"Model path does not exist: {target_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"Loading HeartMuLa Transcriptor from {target_path}...")
        try:
            pipeline = HeartTranscriptorPipeline.from_pretrained(target_path, device=device, dtype=dtype)
            return io.NodeOutput(pipeline)
        except Exception as e:
            raise RuntimeError(f"Failed to load HeartMuLa Transcriptor: {e}")

class HeartMuLaLyricsTranscriber(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HeartMuLaLyricsTranscriber",
            display_name="HeartMuLa Lyrics Transcriber",
            category="HeartMuLa",
            inputs=[
                HEARTMULA_TRANSCRIPTOR.Input("transcriptor"),
                AUDIO.Input("audio"),
                io.Int.Input("max_new_tokens", default=256, min=1, max=445, step=1),
                io.Int.Input("num_beams", default=2, min=1, max=5, step=1),
                io.Boolean.Input("condition_on_prev_tokens", default=False),
                io.Float.Input("logprob_threshold", default=-1.0, min=-20.0, max=0.0, step=0.1),
                io.Float.Input("no_speech_threshold", default=0.4, min=0.0, max=1.0, step=0.01),
                io.Float.Input("temperature", default=0.0, min=0.0, max=1.0, step=0.1),
            ],
            outputs=[io.String.Output()]
        )

    @classmethod
    def execute(cls, transcriptor, audio, max_new_tokens, num_beams, condition_on_prev_tokens, logprob_threshold, no_speech_threshold, temperature) -> io.NodeOutput:
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
            return io.NodeOutput(result["text"])
        finally:
            print("Moving Whisper to CPU...")
            transcriptor.model.to("cpu")
            comfy.model_management.soft_empty_cache()

class HeartMuLaPostProcessor(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HeartMuLaPostProcessor",
            display_name="Audio Post-Processor",
            category="HeartMuLa",
            inputs=[
                AUDIO.Input("audio"),
                io.Boolean.Input("normalize", default=True),
                io.Float.Input("stereo_width", default=1.0, min=0.0, max=2.0, step=0.05),
                io.Int.Input("high_pass", default=80, min=0, max=1000, step=1),
                io.Int.Input("low_pass", default=18000, min=0, max=24000, step=100),
                io.Float.Input("gain_db", default=0.0, min=-20.0, max=20.0, step=0.1),
            ],
            outputs=[AUDIO.Output()]
        )

    @classmethod
    def execute(cls, audio, normalize, stereo_width, high_pass, low_pass, gain_db) -> io.NodeOutput:
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
        return io.NodeOutput({"waveform": waveform.to(device), "sample_rate": sample_rate})
