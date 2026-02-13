from transformers.pipelines.base import Pipeline
from tokenizers import Tokenizer
from ..heartmula.modeling_heartmula import HeartMuLa
from ..heartcodec.modeling_heartcodec import HeartCodec
import torch
import warnings
from typing import Dict, Any, Optional
import os
from dataclasses import dataclass
from tqdm import tqdm
import torchaudio
import json

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.recompile_limit = 128
torch._dynamo.config.force_parameter_static_shapes = False

try:
    import torch._inductor.config as inductor_config
    inductor_config.fallback_random = True
    inductor_config.permute_fusion = False 
    inductor_config.epilogue_fusion = False
    inductor_config.triton.cudagraphs = False 
except ImportError:
    pass

def _get_compile_backend(requested_backend: Optional[str]) -> str:
    if requested_backend:
        return requested_backend
    try:
        import triton
        return "inductor"
    except ImportError:
        return "eager"

@dataclass
class HeartMuLaGenConfig:
    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0

    @classmethod
    def from_file(cls, path: str):
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        return cls(**data)

class HeartMuLaModel:
    def __init__(
        self,
        model: HeartMuLa,
        text_tokenizer: Tokenizer,
        config: HeartMuLaGenConfig,
        dtype: torch.dtype,
        compile_model: bool = False,
        compile_backend: Optional[str] = None,
        compile_mode: str = "default",
    ):
        self.model = model.to(dtype)
        self.text_tokenizer = text_tokenizer
        self.config = config
        self.dtype = dtype
        
        self._compile_model = compile_model
        self._compile_backend = compile_backend
        self._compile_mode = compile_mode
        self._compilation_done = False
        self._cache_bs = 0

        self._parallel_number = 8 + 1
        if hasattr(model.config, "audio_num_codebooks"):
            self._parallel_number = model.config.audio_num_codebooks + 1

    def setup_caches(self, max_batch_size: int):
        if self._cache_bs != max_batch_size:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*Key value caches are already setup.*")
                self.model.setup_caches(max_batch_size=max_batch_size)
            self._cache_bs = max_batch_size
        else:
            self.model.reset_caches()

    def ensure_compiled(self):
        if self._compile_model and not self._compilation_done:
            self._apply_layer_compilation()
            self._compilation_done = True

    def _apply_layer_compilation(self):
        backend = _get_compile_backend(self._compile_backend)
        print(f"Compiling HeartMuLa Transformer layers with backend='{backend}', mode='{self._compile_mode}'...")
        
        def compile_module_layers(module_name, module):
            if hasattr(module, "layers"):
                print(f"  - Compiling {len(module.layers)} individual blocks in {module_name}...")
                for i in range(len(module.layers)):
                    try:
                        module.layers[i] = torch.compile(
                            module.layers[i],
                            backend=backend,
                            mode=self._compile_mode,
                            fullgraph=False,
                            dynamic=False
                        )
                    except Exception as e:
                        print(f"      ! Block {i} compilation failed: {e}")
            else:
                 warnings.warn(f"Could not find 'layers' in {module_name} to compile.")

        compile_module_layers("Backbone", self.model.backbone)
        compile_module_layers("Internal Decoder", self.model.decoder)
        print("HeartMuLa compilation setup complete.")

    def preprocess(self, inputs: Dict[str, Any], cfg_scale: float):
        tags = inputs["tags"]
        if os.path.isfile(tags):
            with open(tags, encoding="utf-8") as fp:
                tags = fp.read()
        assert isinstance(tags, str), f"tags must be a string, but got {type(tags)}"

        tags = tags.lower()
        if not tags.startswith("<tag>"):
            tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"):
            tags = f"{tags}</tag>"

        tags_ids = self.text_tokenizer.encode(tags).ids
        if tags_ids[0] != self.config.text_bos_id:
            tags_ids = [self.config.text_bos_id] + tags_ids
        if tags_ids[-1] != self.config.text_eos_id:
            tags_ids = tags_ids + [self.config.text_eos_id]

        muq_embed = torch.zeros([self.model.config.muq_dim], dtype=self.dtype)
        muq_idx = len(tags_ids)

        lyrics = inputs["lyrics"]
        if os.path.isfile(lyrics):
            with open(lyrics, encoding="utf-8") as fp:
                lyrics = fp.read()
        assert isinstance(lyrics, str), f"lyrics must be a string, but got {type(lyrics)}"
        lyrics = lyrics.lower()

        lyrics_ids = self.text_tokenizer.encode(lyrics).ids
        if lyrics_ids[0] != self.config.text_bos_id:
            lyrics_ids = [self.config.text_bos_id] + lyrics_ids
        if lyrics_ids[-1] != self.config.text_eos_id:
            lyrics_ids = lyrics_ids + [self.config.text_eos_id]

        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
        tokens = torch.zeros([prompt_len, self._parallel_number], dtype=torch.long)
        tokens[: len(tags_ids), -1] = torch.tensor(tags_ids)
        tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids)

        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens_mask[:, -1] = True

        bs_size = 2 if cfg_scale != 1.0 else 1

        def _cfg_cat(tensor: torch.Tensor, cfg_scale: float):
            tensor = tensor.unsqueeze(0)
            if cfg_scale != 1.0:
                tensor = torch.cat([tensor, tensor], dim=0)
            return tensor

        return {
            "tokens": _cfg_cat(tokens, cfg_scale),
            "tokens_mask": _cfg_cat(tokens_mask, cfg_scale),
            "muq_embed": _cfg_cat(muq_embed, cfg_scale),
            "muq_idx": [muq_idx] * bs_size,
            "pos": _cfg_cat(torch.arange(prompt_len, dtype=torch.long), cfg_scale),
        }

    def generate_tokens(
        self,
        processed_inputs: Dict[str, Any],
        max_audio_length_ms: int,
        temperature: float,
        topk: int,
        cfg_scale: float,
        callback: Optional[Any] = None,
    ):
        prompt_tokens = processed_inputs["tokens"]
        prompt_tokens_mask = processed_inputs["tokens_mask"]
        continuous_segment = processed_inputs["muq_embed"]
        starts = processed_inputs["muq_idx"]
        prompt_pos = processed_inputs["pos"]

        frames = []
        device = prompt_tokens.device

        with torch.autocast(device_type=device.type, dtype=self.dtype):
            curr_token = self.model.generate_frame(
                tokens=prompt_tokens,
                tokens_mask=prompt_tokens_mask,
                input_pos=prompt_pos,
                temperature=temperature,
                topk=topk,
                cfg_scale=cfg_scale,
                continuous_segments=continuous_segment,
                starts=starts,
            )
        frames.append(curr_token[0:1,])

        def _pad_audio_token(token: torch.Tensor):
            padded_token = (
                torch.ones(
                    (token.shape[0], self._parallel_number),
                    device=token.device,
                    dtype=torch.long,
                )
                * self.config.empty_id
            )
            padded_token[:, :-1] = token
            padded_token = padded_token.unsqueeze(1)
            padded_token_mask = torch.ones_like(
                padded_token, device=token.device, dtype=torch.bool
            )
            padded_token_mask[..., -1] = False
            return padded_token, padded_token_mask

        max_audio_frames = max_audio_length_ms // 80

        for i in tqdm(range(max_audio_frames)):
            if callback:
                callback(1)
            curr_token, curr_token_mask = _pad_audio_token(curr_token)
            
            with torch.autocast(device_type=device.type, dtype=self.dtype):
                curr_token = self.model.generate_frame(
                    tokens=curr_token,
                    tokens_mask=curr_token_mask,
                    input_pos=prompt_pos[..., -1:] + i + 1,
                    temperature=temperature,
                    topk=topk,
                    cfg_scale=cfg_scale,
                    continuous_segments=None,
                    starts=None,
                )
            
            if torch.any(curr_token[0:1, :] >= self.config.audio_eos_id):
                break
            frames.append(curr_token[0:1,])
            
        frames = torch.stack(frames).permute(1, 2, 0).squeeze(0)
        return frames

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        dtype: torch.dtype,
        compile_model: bool = False,
        compile_backend: Optional[str] = None,
        compile_mode: str = "default",
    ):
        model = HeartMuLa.from_pretrained(
            model_path, 
            dtype=dtype, 
            quantization_config=None, 
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True
        )
        
        base_dir = os.path.dirname(model_path)
        
        tokenizer_path = os.path.join(base_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
            tokenizer_path = os.path.join(model_path, "tokenizer.json")
            
        gen_config_path = os.path.join(base_dir, "gen_config.json")
        if not os.path.exists(gen_config_path):
            gen_config_path = os.path.join(model_path, "gen_config.json")

        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_path}")
        if not os.path.isfile(gen_config_path):
            raise FileNotFoundError(f"gen_config.json not found at {gen_config_path}")

        tokenizer = Tokenizer.from_file(tokenizer_path)
        gen_config = HeartMuLaGenConfig.from_file(gen_config_path)

        return cls(model, tokenizer, gen_config, dtype, compile_model, compile_backend, compile_mode)

class HeartCodecModel:
    def __init__(
        self,
        audio_codec: HeartCodec,
    ):
        self.audio_codec = audio_codec

    def decode_tokens(self, frames, duration=29.76, callback=None):
        wav = self.audio_codec.detokenize(frames, duration=duration, callback=callback)
        return {"wav": wav}

    @classmethod
    def from_pretrained(
        cls,
        codec_path: str,
    ):
        audio_codec = HeartCodec.from_pretrained(
            codec_path, 
            device_map="cpu", 
            dtype=torch.float32,
            ignore_mismatched_sizes=True
        )
        return cls(audio_codec)