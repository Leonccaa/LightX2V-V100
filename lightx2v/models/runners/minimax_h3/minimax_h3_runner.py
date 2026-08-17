import json
import os
import time
from contextlib import suppress

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageOps
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from lightx2v.models.networks.minimax_h3.lora import MiniMaxH3LoraAdapter
from lightx2v.models.networks.minimax_h3.model import MiniMaxH3Model
from lightx2v.models.networks.minimax_h3.packing import (
    TEXT_TAG,
    align_num_frames,
    prepare_keyframe_image,
    resolve_canvas_size,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    validate_t2av_geometry,
)
from lightx2v.models.networks.minimax_h3.packing_ref2av import (
    MAX_REFERENCES,
    MAX_REFERENCE_AUDIOS,
    MAX_REFERENCE_IMAGES,
    MAX_REFERENCE_VIDEOS,
    MiniMaxH3PreparedReference,
    decode_reference_audio,
    decode_reference_video,
    prepare_reference_frames,
    prepare_reference_image,
    prepare_reference_waveform,
    resample_reference_frames,
    resolve_reference_image_size,
    trim_reference_num_frames,
)
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.minimax_h3 import MiniMaxH3Scheduler
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import DTYPE_MAP, GET_RECORDER_MODE
from lightx2v.utils.input_info import FL2AVInputInfo, I2AVInputInfo, L2AVInputInfo, Ref2AVInputInfo, T2AVInputInfo
from lightx2v.utils.profiler import ProfilingContext4DebugL1, ProfilingContext4DebugL2
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE

torch_device_module = getattr(torch, AI_DEVICE)


def _summarize_checkpoint_tensors(tensors):
    logical_bytes = 0
    unique_storage_bytes = 0
    seen_storages = set()
    by_dtype = {}
    by_device = {}
    for tensor in tensors.values():
        tensor_bytes = tensor.numel() * tensor.element_size()
        logical_bytes += tensor_bytes
        dtype_bucket = by_dtype.setdefault(str(tensor.dtype), {"tensors": 0, "logical_bytes": 0})
        dtype_bucket["tensors"] += 1
        dtype_bucket["logical_bytes"] += tensor_bytes
        device_bucket = by_device.setdefault(str(tensor.device), {"tensors": 0, "logical_bytes": 0})
        device_bucket["tensors"] += 1
        device_bucket["logical_bytes"] += tensor_bytes
        storage = tensor.untyped_storage()
        storage_key = (str(tensor.device), storage.data_ptr())
        if storage_key not in seen_storages:
            seen_storages.add(storage_key)
            unique_storage_bytes += storage.nbytes()
    return {
        "tensors": len(tensors),
        "logical_bytes": logical_bytes,
        "unique_storage_bytes": unique_storage_bytes,
        "by_dtype": by_dtype,
        "by_device": by_device,
    }


def build_minimax_h3_model_with_lora(config, model_kwargs, lora_configs):
    """Build H3 with either a dynamic LoRA branch or load-time merging."""
    if config.get("lora_dynamic_apply", False):
        if len(lora_configs) != 1:
            raise ValueError("MiniMax-H3 dynamic LoRA currently accepts exactly one lora_configs entry")
        lora_config = lora_configs[0]
        if not lora_config.get("path"):
            raise ValueError("MiniMax-H3 dynamic LoRA requires lora_configs[0].path")
        if lora_config.get("alpha") is None:
            raise ValueError("MiniMax-H3 dynamic LoRA requires lora_configs[0].alpha (use 8 for the MiniMax-H3 Turbo LoRA)")
        model_kwargs.update(
            lora_path=lora_config["path"],
            lora_strength=lora_config.get("strength", 1.0),
            lora_alpha=lora_config["alpha"],
        )
        return MiniMaxH3Model(**model_kwargs)
    if config.get("dit_quantized", False):
        raise ValueError("MiniMax-H3 merged LoRA inference requires original, non-quantized DiT weights")
    if config.get("lazy_load", False):
        raise ValueError("MiniMax-H3 lazy loading does not support LoRA merging")

    if not config.get("h3_benchmark_load_telemetry", False):
        model = MiniMaxH3Model(**model_kwargs)
        MiniMaxH3LoraAdapter(model).apply_lora(lora_configs)
        return model

    base_started_epoch = time.time()
    base_started = time.perf_counter()
    model = MiniMaxH3Model(**model_kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    base_finished_epoch = time.time()
    base_seconds = time.perf_counter() - base_started
    checkpoint_tensor_inventory = _summarize_checkpoint_tensors(model.original_weight_dict)
    base_memory = {
        "allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
        "reserved_bytes": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    lora_started_epoch = time.time()
    lora_started = time.perf_counter()
    MiniMaxH3LoraAdapter(model).apply_lora(lora_configs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    lora_finished_epoch = time.time()
    model.h3_load_telemetry = {
        "base_checkpoint_and_model": {
            "started_epoch": base_started_epoch,
            "finished_epoch": base_finished_epoch,
            "elapsed_seconds": base_seconds,
            "cuda_memory": base_memory,
            "checkpoint_tensor_inventory": checkpoint_tensor_inventory,
        },
        "lora_merge": {
            "started_epoch": lora_started_epoch,
            "finished_epoch": lora_finished_epoch,
            "elapsed_seconds": time.perf_counter() - lora_started,
            "cuda_memory": {
                "allocated_bytes": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "reserved_bytes": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
            },
        },
    }
    return model


@RUNNER_REGISTER("minimax_h3")
class MiniMaxH3Runner(DefaultRunner):
    """Native MiniMax-H3 audio-video runner.

    Transformer, text-encoder, and VAE residency are configured independently.
    ``cpu_offload`` controls only the transformer, while
    ``text_encoder_cpu_offload`` and ``vae_cpu_offload`` control the native
    Qwen3-VL conditioner and both native VAEs. This mirrors Wan's component
    offload behavior while keeping Diffusers out of the runtime dependency
    graph.
    """

    _WARMUP_SHAPES = (
        (480, 480, 158),  # aligned from a 6-second request
        (544, 960, 124),
    )
    _WARMUP_STEP_COUNT = 2
    _WARMUP_TASKS = ("t2av", "fl2av", "i2av", "l2av", "ref2av")

    def __init__(self, config):
        if config.get("task") not in {"t2av", "i2av", "l2av", "fl2av", "ref2av"}:
            raise ValueError("MiniMax-H3 supports t2av/i2av/l2av/fl2av/ref2av")
        self.loaded_transformer_partition = "transformer_ref" if config["task"] == "ref2av" else "transformer"
        if config.get("lazy_load", False) or config.get("unload_modules", False):
            raise NotImplementedError("MiniMax-H3 does not support lazy_load or unload_modules yet; use the released sharded checkpoint with model or block CPU offload.")
        super().__init__(config)

    def init_modules(self):
        super().init_modules()
        self.run_input_encoder = self._run_input_encoder_local_h3

    @ProfilingContext4DebugL1("Warmup")
    def run_warmup(self):
        task = self.config["task"]
        if task not in self._WARMUP_TASKS:
            raise NotImplementedError(f"MiniMax-H3 warmup does not support task: {task}")

        for height, width, num_frames in self._WARMUP_SHAPES:
            logger.info(f"Warmup: {height}x{width}x{num_frames}")
            transformer_offloaded = not self.config.get("cpu_offload", False)
            try:
                self.scheduler.generator = None
                self._prepare_warmup_inputs(height, width, num_frames)
                self.inputs = self._run_input_encoder_local_h3()
                self.init_run()

                for step_index in range(min(self._WARMUP_STEP_COUNT, self.scheduler.infer_steps)):
                    self.scheduler.step_pre(step_index)
                    self.model.infer(self.inputs)
                    self.scheduler.step_post()
                video_rows = self.scheduler.video_latents
                audio_rows = self.scheduler.audio_latents

                if self.config.get("cpu_offload", False):
                    self._offload_transformer()
                    transformer_offloaded = True

                self.run_vae_decoder(video_rows, audio_rows)
                torch_device_module.synchronize()
            finally:
                if self.config.get("cpu_offload", False) and not transformer_offloaded:
                    with suppress(Exception):
                        self._offload_transformer()
                self.clear_warmup_state()

        logger.info("[Warmup] Warmup completed")
        self._maybe_freeze_gc()

    def _prepare_warmup_inputs(self, height, width, num_frames):
        task = self.config["task"]
        common = {
            "seed": 0,
            "prompt": "A sunrise over distant mountains reflected across a calm lake beneath drifting clouds."
            if (height, width, num_frames) == self._WARMUP_SHAPES[0]
            else "A cinematic fox walking through a snowy forest.",
            "target_shape": [height, width],
            "target_video_length": num_frames,
            "return_result_tensor": True,
        }
        image = Image.new("RGB", (width, height), color=0)
        if task == "t2av":
            self.input_info = T2AVInputInfo(**common)
        elif task == "i2av":
            self.input_info = I2AVInputInfo(**common, image_path=image)
        elif task == "l2av":
            self.input_info = L2AVInputInfo(**common, last_frame_path=image)
        elif task == "fl2av":
            self.input_info = FL2AVInputInfo(**common, image_path=image, last_frame_path=image.copy())
        else:
            self.input_info = Ref2AVInputInfo(**common, image_path=image)

    def clear_warmup_state(self):
        self.scheduler.clear()
        self.condition_video_latents = []
        self.condition_audio_latents = []
        self.keyframe_anchors = ()
        self.prepared_references = None
        self.input_info = None
        self.__dict__.pop("inputs", None)

    def init_scheduler(self):
        self.scheduler = MiniMaxH3Scheduler(self.config)

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = [] if self.config.get("precomputed_condition_path") else self.load_text_encoder()
        if self.config.get("h3_dit_only", False):
            self.video_vae, self.audio_vae = None, None
        else:
            self.video_vae, self.audio_vae = self.load_vae()

    def load_transformer(self):
        model_kwargs = {
            "model_path": self.config["model_path"],
            "config": self.config,
            "device": self.init_device,
        }
        lora_configs = self.config.get("lora_configs")
        if lora_configs:
            return build_minimax_h3_model_with_lora(
                self.config,
                model_kwargs,
                lora_configs,
            )
        return MiniMaxH3Model(**model_kwargs)

    def load_text_encoder(self):
        from lightx2v.models.input_encoders.hf.minimax_h3 import MiniMaxH3Qwen3VLTextEncoder

        return [MiniMaxH3Qwen3VLTextEncoder(self.config)]

    def load_vae(self):
        from lightx2v.models.audio_encoders.hf.minimax_h3 import MiniMaxH3AudioVAE
        from lightx2v.models.video_encoders.hf.minimax_h3 import MiniMaxH3VideoVAE

        cpu_offload = self.config.get("vae_cpu_offload", self.config.get("cpu_offload", False))
        video_vae_quantized = self.config.get("video_vae_quantized", False)
        video_vae_quant_scheme = self.config["video_vae_quant_scheme"] if video_vae_quantized else None
        video_vae_quantized_ckpt = self.config["video_vae_quantized_ckpt"] if video_vae_quantized else None
        vae_sensitive_layer_dtype = DTYPE_MAP[self.config.get("vae_sensitive_layer_dtype", "fp32")]
        video_vae = MiniMaxH3VideoVAE.from_pretrained(
            self.config["model_path"],
            device=AI_DEVICE,
            cpu_offload=cpu_offload,
            checkpoint_path=video_vae_quantized_ckpt,
            quant_scheme=video_vae_quant_scheme,
            sensitive_layer_dtype=vae_sensitive_layer_dtype,
            use_compile=self.config.get("vae_use_compile", False),
            attn_type=self.config.get("vae_attn_type", "torch_sdpa"),
        )
        if self.config.get("vae_decode_parallel", False):
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            if world_size > 1:
                video_vae.enable_decode_parallel()
                logger.info(f"MiniMax-H3 spatial-tile VAE decode parallel enabled over {world_size} ranks")
            else:
                logger.info("MiniMax-H3 spatial-tile VAE decode parallel disabled for single-rank inference")
        audio_vae = MiniMaxH3AudioVAE.from_pretrained(self.config["model_path"], device=AI_DEVICE, cpu_offload=cpu_offload)
        configured_sample_rate = int(self.config.get("audio_sampling_rate", audio_vae.sampling_rate))
        if configured_sample_rate != audio_vae.sampling_rate:
            raise ValueError(f"MiniMax-H3 audio_sampling_rate must match the Audio VAE checkpoint: config={configured_sample_rate}, checkpoint={audio_vae.sampling_rate}")
        return video_vae, audio_vae

    def _resolve_request_geometry(self, geometry_image=None):
        if self.input_info.target_shape:
            if len(self.input_info.target_shape) != 2:
                raise ValueError(f"MiniMax-H3 target_shape must be [height, width], got {self.input_info.target_shape}")
            height, width = (int(value) for value in self.input_info.target_shape)
        elif geometry_image is not None:
            height, width = resolve_canvas_size(*geometry_image.size)
            self.input_info.target_shape = [height, width]
        else:
            height = int(self.config["target_height"])
            width = int(self.config["target_width"])
            self.input_info.target_shape = [height, width]

        requested_frames = int(self.input_info.target_video_length or self.config.get("target_video_length", 124))
        num_frames = align_num_frames(requested_frames)
        if num_frames != requested_frames:
            logger.warning(f"MiniMax-H3 frame count must be 17*n+5; aligning {requested_frames} upward to {num_frames}")
        validate_t2av_geometry(num_frames, height, width)
        self.input_info.target_video_length = num_frames
        self.request_height = height
        self.request_width = width
        self.request_num_frames = num_frames

    @ProfilingContext4DebugL1(
        "Run Text Encoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_text_encode_duration,
        metrics_labels=["MiniMaxH3Runner"],
    )
    def run_text_encoder(self, input_info, keyframes=None, references=None):
        negative_prompt = (input_info.negative_prompt or "").strip()
        if negative_prompt:
            logger.warning("MiniMax-H3 is guidance-distilled; negative_prompt is ignored")
        precomputed_path = self.config.get("precomputed_condition_path")
        if precomputed_path:
            if keyframes or references:
                raise ValueError("MiniMax-H3 precomputed conditioning currently supports t2av without image/reference rows")
            tensors = load_file(precomputed_path, device="cpu")
            required = {"prompt_embeds", "text_token_tags"}
            missing = required.difference(tensors)
            if missing:
                raise ValueError(f"MiniMax-H3 precomputed conditioning is missing tensors: {sorted(missing)}")
            prompt_embeds = tensors["prompt_embeds"]
            text_token_tags = tensors["text_token_tags"]
            if prompt_embeds.ndim != 2 or prompt_embeds.shape[-1] != 5120:
                raise ValueError(f"MiniMax-H3 prompt_embeds must be [tokens, 5120], got {tuple(prompt_embeds.shape)}")
            if text_token_tags.ndim != 1 or text_token_tags.shape[0] != prompt_embeds.shape[0]:
                raise ValueError(
                    "MiniMax-H3 text_token_tags must be one-dimensional and match prompt_embeds rows; "
                    f"got embeds={tuple(prompt_embeds.shape)}, tags={tuple(text_token_tags.shape)}"
                )
            logger.info(f"Loaded precomputed MiniMax-H3 conditioning from {precomputed_path}: {tuple(prompt_embeds.shape)}")
            return {
                "prompt_embeds": prompt_embeds.to(AI_DEVICE),
                "text_token_tags": text_token_tags.to(AI_DEVICE),
            }
        return self.text_encoders[0].infer(input_info.prompt, image_list=keyframes, references=references)

    def _load_precomputed_conditioning_bundle(self):
        precomputed_path = self.config.get("precomputed_condition_path")
        if not precomputed_path:
            raise ValueError("MiniMax-H3 precomputed conditioning path is not configured")
        with safe_open(precomputed_path, framework="pt", device="cpu") as source:
            metadata = dict(source.metadata() or {})
            tensors = {key: source.get_tensor(key) for key in source.keys()}
        required = {"prompt_embeds", "text_token_tags"}
        missing = required.difference(tensors)
        if missing:
            raise ValueError(f"MiniMax-H3 precomputed conditioning is missing tensors: {sorted(missing)}")
        prompt_embeds = tensors["prompt_embeds"]
        text_token_tags = tensors["text_token_tags"]
        if prompt_embeds.ndim != 2 or prompt_embeds.shape[-1] != 5120:
            raise ValueError(f"MiniMax-H3 prompt_embeds must be [tokens, 5120], got {tuple(prompt_embeds.shape)}")
        if text_token_tags.ndim != 1 or text_token_tags.shape[0] != prompt_embeds.shape[0]:
            raise ValueError(
                "MiniMax-H3 text_token_tags must be one-dimensional and match prompt_embeds rows; "
                f"got embeds={tuple(prompt_embeds.shape)}, tags={tuple(text_token_tags.shape)}"
            )
        raw_bundle = metadata.get("minimax_h3_bundle")
        bundle = json.loads(raw_bundle) if raw_bundle else {
            "format": "minimax_h3_conditioning_bundle",
            "version": 0,
            "task": "t2av",
            "keyframes": [],
            "references": [],
        }
        if bundle.get("format") != "minimax_h3_conditioning_bundle":
            raise ValueError(f"Unsupported MiniMax-H3 conditioning bundle format: {bundle.get('format')!r}")
        if int(bundle.get("version", 0)) not in {0, 1}:
            raise ValueError(f"Unsupported MiniMax-H3 conditioning bundle version: {bundle.get('version')!r}")
        logger.info(
            "Loaded precomputed MiniMax-H3 {} conditioning from {}: {}, keyframes={}, references={}",
            bundle.get("task", "t2av"),
            precomputed_path,
            tuple(prompt_embeds.shape),
            len(bundle.get("keyframes") or []),
            len(bundle.get("references") or []),
        )
        return (
            {
                "prompt_embeds": prompt_embeds.to(AI_DEVICE),
                "text_token_tags": text_token_tags.to(AI_DEVICE),
            },
            bundle,
            tensors,
        )

    @staticmethod
    def _bundle_tensor(tensors, name, description):
        if not name or name not in tensors:
            raise ValueError(f"MiniMax-H3 conditioning bundle is missing {description} tensor {name!r}")
        return tensors[name]

    def _restore_precomputed_media_conditioning(self, bundle, tensors):
        task = self.config["task"]
        bundle_task = bundle.get("task", "t2av")
        if bundle_task != task:
            raise ValueError(f"MiniMax-H3 conditioning task mismatch: runner={task!r}, bundle={bundle_task!r}")
        frame_count = bundle.get("frame_count")
        if frame_count is not None and int(frame_count) != self.request_num_frames:
            raise ValueError(
                f"MiniMax-H3 conditioning frame mismatch: request={self.request_num_frames}, bundle={frame_count}"
            )

        if task in {"i2av", "l2av", "fl2av"}:
            entries = bundle.get("keyframes") or []
            expected = {"i2av": ("first",), "l2av": ("last",), "fl2av": ("first", "last")}[task]
            anchors = tuple(str(entry.get("anchor")) for entry in entries)
            if anchors != expected:
                raise ValueError(f"MiniMax-H3 {task} bundle needs anchors {expected}, got {anchors}")
            for index, entry in enumerate(entries):
                latent = self._bundle_tensor(tensors, entry.get("tensor"), f"keyframe {index}")
                if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != int(self.config.get("in_channels", 24)):
                    raise ValueError(f"MiniMax-H3 keyframe {index} latent must be [1,24,T,H,W], got {tuple(latent.shape)}")
                if tuple(latent.shape[-2:]) != (
                    self.request_height // int(self.config.get("vae_spatial_scale_factor", 16)),
                    self.request_width // int(self.config.get("vae_spatial_scale_factor", 16)),
                ):
                    raise ValueError(
                        f"MiniMax-H3 keyframe {index} latent geometry {tuple(latent.shape[-2:])} does not match "
                        f"request {self.request_height}x{self.request_width}"
                    )
                self.condition_video_latents.append(latent)
            self.keyframe_anchors = anchors
            return

        if task == "ref2av":
            entries = bundle.get("references") or []
            if not entries:
                raise ValueError("MiniMax-H3 ref2av conditioning bundle has no references")
            references = []
            for index, entry in enumerate(entries):
                kind = str(entry.get("kind", ""))
                if kind not in {"image", "video", "audio"}:
                    raise ValueError(f"Unsupported MiniMax-H3 reference kind at index {index}: {kind!r}")
                has_audio = bool(entry.get("has_audio", False))
                reference = MiniMaxH3PreparedReference(kind=kind, has_audio=has_audio)
                if kind != "audio":
                    latent = self._bundle_tensor(tensors, entry.get("video_tensor"), f"reference {index} video")
                    if latent.ndim != 5 or latent.shape[0] != 1 or latent.shape[1] != int(self.config.get("in_channels", 24)):
                        raise ValueError(f"MiniMax-H3 reference {index} video latent must be [1,24,T,H,W], got {tuple(latent.shape)}")
                    reference.num_latent_frames = int(entry.get("num_latent_frames", latent.shape[2]))
                    reference.latent_height = int(entry.get("latent_height", latent.shape[3]))
                    reference.latent_width = int(entry.get("latent_width", latent.shape[4]))
                    if (reference.num_latent_frames, reference.latent_height, reference.latent_width) != tuple(latent.shape[2:]):
                        raise ValueError(
                            f"MiniMax-H3 reference {index} metadata/latent shape mismatch: "
                            f"metadata={(reference.num_latent_frames, reference.latent_height, reference.latent_width)}, "
                            f"latent={tuple(latent.shape[2:])}"
                        )
                    self.condition_video_latents.append(latent)
                if has_audio:
                    audio = self._bundle_tensor(tensors, entry.get("audio_tensor"), f"reference {index} audio")
                    if audio.ndim == 4 and audio.shape[0] == 1:
                        audio = audio.squeeze(0).permute(1, 0, 2).contiguous()
                    if audio.ndim != 3 or audio.shape[0] != 2 or audio.shape[1] != int(self.config.get("audio_in_channels", 32)):
                        raise ValueError(f"MiniMax-H3 reference {index} audio latent must be [2,32,T], got {tuple(audio.shape)}")
                    reference.num_audio_latents = int(entry.get("num_audio_latents", audio.shape[-1]))
                    if reference.num_audio_latents != audio.shape[-1]:
                        raise ValueError(
                            f"MiniMax-H3 reference {index} audio metadata/latent length mismatch: "
                            f"metadata={reference.num_audio_latents}, latent={audio.shape[-1]}"
                        )
                    self.condition_audio_latents.append(audio)
                references.append(reference)
            self.prepared_references = references
            return

        if task != "t2av":
            raise ValueError(f"MiniMax-H3 precomputed media restoration does not support task {task!r}")
        if bundle.get("keyframes") or bundle.get("references"):
            raise ValueError("MiniMax-H3 t2av conditioning bundle must not contain media conditions")

    @staticmethod
    def _load_rgb_image(value):
        if isinstance(value, Image.Image):
            image = value
        else:
            image = Image.open(value)
        return ImageOps.exif_transpose(image).convert("RGB")

    def _prepare_keyframes(self):
        task = self.config["task"]
        if task == "t2av":
            if not isinstance(self.input_info, T2AVInputInfo):
                raise TypeError(f"MiniMax-H3 t2av expects T2AVInputInfo, got {type(self.input_info).__name__}")
            return [], ()
        if task == "i2av":
            if not isinstance(self.input_info, I2AVInputInfo) or not self.input_info.image_path:
                raise ValueError("MiniMax-H3 i2av requires exactly one --image_path")
            values, anchors = [self.input_info.image_path], ("first",)
        elif task == "l2av":
            if not isinstance(self.input_info, L2AVInputInfo) or not self.input_info.last_frame_path:
                raise ValueError("MiniMax-H3 l2av requires --last_frame_path")
            values, anchors = [self.input_info.last_frame_path], ("last",)
        elif task == "fl2av":
            if not isinstance(self.input_info, FL2AVInputInfo) or not self.input_info.image_path or not self.input_info.last_frame_path:
                raise ValueError("MiniMax-H3 fl2av requires --image_path and --last_frame_path")
            values, anchors = [self.input_info.image_path, self.input_info.last_frame_path], ("first", "last")
        else:
            return [], ()
        if any(isinstance(value, str) and "," in value for value in values):
            raise ValueError(f"MiniMax-H3 {task} accepts one file per frame argument, not comma-separated lists")
        images = [self._load_rgb_image(value) for value in values]
        self._resolve_request_geometry(images[0])
        images = [prepare_keyframe_image(image, self.request_height, self.request_width, stretch=index == 0) for index, image in enumerate(images)]
        return images, anchors

    @staticmethod
    def _split_reference_paths(value):
        """Normalize the shared media CLI fields into individual references."""
        if value is None:
            return []
        if isinstance(value, str):
            return [path.strip() for path in value.split(",") if path.strip()]
        if isinstance(value, (list, tuple)):
            paths = []
            for path in value:
                if path is None:
                    continue
                if isinstance(path, str):
                    path = path.strip()
                    if not path:
                        continue
                paths.append(path)
            return paths
        return [value]

    def _prepare_references(self):
        from lightx2v.models.video_encoders.hf.ltx2.audio_vae.ops import Audio

        if not isinstance(self.input_info, Ref2AVInputInfo):
            raise TypeError(f"MiniMax-H3 ref2av expects Ref2AVInputInfo, got {type(self.input_info).__name__}")
        entries = []
        for kind, value in (
            ("image", self.input_info.image_path),
            ("video", self.input_info.video_path),
            ("audio", self.input_info.audio_path),
        ):
            entries.extend({kind: path} for path in self._split_reference_paths(value))
        if not entries:
            raise ValueError("MiniMax-H3 ref2av requires --image_path, --video_path, or --audio_path")
        if len(entries) > MAX_REFERENCES:
            raise ValueError(f"MiniMax-H3 ref2av accepts at most {MAX_REFERENCES} references")
        kinds = ["image" if "image" in entry else "video" if "video" in entry else "audio" for entry in entries]
        if kinds.count("image") > MAX_REFERENCE_IMAGES or kinds.count("video") > MAX_REFERENCE_VIDEOS:
            raise ValueError("MiniMax-H3 ref2av reference image/video count exceeds 9/3")
        if all(kind == "audio" for kind in kinds):
            raise ValueError("MiniMax-H3 ref2av does not allow audio-only references")

        references = []
        audio_count = 0
        max_duration = self.request_num_frames / 24.0
        for entry, kind in zip(entries, kinds):
            if kind == "image":
                image = self._load_rgb_image(entry["image"])
                height, width = resolve_reference_image_size(*image.size)
                references.append(MiniMaxH3PreparedReference("image", image=prepare_reference_image(image, height, width)))
                continue
            if kind == "video":
                video = entry["video"]
                soundtrack = None
                if isinstance(video, (str, os.PathLike)):
                    frames, fps, decoded_soundtrack = decode_reference_video(video)
                    if decoded_soundtrack is not None:
                        waveform, sample_rate = decoded_soundtrack
                        soundtrack = Audio(
                            waveform=waveform,
                            sampling_rate=int(entry.get("sample_rate", sample_rate)),
                        )
                else:
                    frames = np.asarray(video)
                    fps = float(entry.get("fps", 24.0))
                frames = prepare_reference_frames(resample_reference_frames(frames, float(entry.get("fps", fps))), self.request_num_frames)
                reference = MiniMaxH3PreparedReference("video", has_audio=soundtrack is not None, frames=frames)
                if soundtrack is not None:
                    audio_count += 1
                    waveform = soundtrack.waveform.squeeze(0) if soundtrack.waveform.ndim == 3 else soundtrack.waveform
                    reference.waveform = prepare_reference_waveform(waveform, soundtrack.sampling_rate, self.audio_vae.sampling_rate, max_duration)
                references.append(reference)
                continue
            audio_count += 1
            value = entry["audio"]
            if isinstance(value, (str, os.PathLike)):
                waveform, sample_rate = decode_reference_audio(value)
                decoded = Audio(
                    waveform=waveform,
                    sampling_rate=int(entry.get("sample_rate", sample_rate)),
                )
            else:
                decoded = Audio(
                    waveform=torch.as_tensor(value),
                    sampling_rate=int(entry.get("sample_rate", self.audio_vae.sampling_rate)),
                )
            waveform = decoded.waveform.squeeze(0) if decoded.waveform.ndim == 3 else decoded.waveform
            references.append(
                MiniMaxH3PreparedReference(
                    "audio",
                    has_audio=True,
                    waveform=prepare_reference_waveform(waveform, decoded.sampling_rate, self.audio_vae.sampling_rate, max_duration),
                )
            )
        if audio_count > MAX_REFERENCE_AUDIOS:
            raise ValueError(f"MiniMax-H3 ref2av accepts at most {MAX_REFERENCE_AUDIOS} audio-bearing references")
        return references

    def _encode_keyframes(self, keyframes):
        latents = []
        for image in keyframes:
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None, :, None].float().div_(255.0)
            latents.append(self.video_vae.encode_condition(pixels, video=False))
        return latents

    def _encode_references(self, references):
        video_latents, audio_latents = [], []
        for reference in references:
            if reference.kind != "audio":
                if reference.kind == "image":
                    pixels = torch.from_numpy(np.asarray(reference.image).copy()).permute(2, 0, 1)[None, :, None].float().div_(255.0)
                    latent = self.video_vae.encode_condition(pixels, video=False)
                else:
                    frames = reference.frames[: trim_reference_num_frames(reference.frames.shape[0])]
                    pixels = torch.from_numpy(frames.copy()).permute(3, 0, 1, 2)[None].float().div_(255.0)
                    latent = self.video_vae.encode_condition(pixels, video=True)
                reference.num_latent_frames = latent.shape[2]
                reference.latent_height, reference.latent_width = latent.shape[3:]
                video_latents.append(latent)
            if reference.has_audio:
                latent = self.audio_vae.encode(reference.waveform)
                reference.num_audio_latents = latent.shape[-1]
                audio_latents.append(latent)
        return video_latents, audio_latents

    @ProfilingContext4DebugL2("Run Input Encoder")
    def _run_input_encoder_local_h3(self):
        task = self.config["task"]
        requested_partition = "transformer_ref" if task == "ref2av" else "transformer"
        if requested_partition != self.loaded_transformer_partition:
            raise ValueError(
                "MiniMax-H3 cannot switch between the base and reference transformer partitions after initialization; "
                f"loaded {self.loaded_transformer_partition!r}, requested {requested_partition!r}. "
                "Create a separate LightX2VPipeline for ref2av."
            )
        self.condition_video_latents = []
        self.condition_audio_latents = []
        self.keyframe_anchors = ()
        self.prepared_references = None
        if self.config.get("precomputed_condition_path"):
            self._resolve_request_geometry()
            text_encoder_output, bundle, tensors = self._load_precomputed_conditioning_bundle()
            self._restore_precomputed_media_conditioning(bundle, tensors)
        elif task == "ref2av":
            self._resolve_request_geometry()
            self.prepared_references = self._prepare_references()
            text_encoder_output = self.run_text_encoder(self.input_info, references=self.prepared_references)
            with ProfilingContext4DebugL1(
                "Run VAE Encoder",
                recorder_mode=GET_RECORDER_MODE(),
                metrics_func=monitor_cli.lightx2v_run_vae_encoder_image_duration,
                metrics_labels=["MiniMaxH3Runner"],
            ):
                self.condition_video_latents, self.condition_audio_latents = self._encode_references(self.prepared_references)
        else:
            keyframes, self.keyframe_anchors = self._prepare_keyframes()
            if task == "t2av":
                self._resolve_request_geometry()
            text_encoder_output = self.run_text_encoder(self.input_info, keyframes=keyframes)
            if keyframes:
                with ProfilingContext4DebugL1(
                    "Run VAE Encoder",
                    recorder_mode=GET_RECORDER_MODE(),
                    metrics_func=monitor_cli.lightx2v_run_vae_encoder_image_duration,
                    metrics_labels=["MiniMaxH3Runner"],
                ):
                    self.condition_video_latents = self._encode_keyframes(keyframes)
        tags = text_encoder_output["text_token_tags"]
        if tags.ndim != 1:
            raise ValueError("MiniMax-H3 conditioner token tags must be one-dimensional")
        if task == "t2av" and not bool((tags == TEXT_TAG).all()):
            raise ValueError("MiniMax-H3 t2av conditioner returned non-text modality rows")
        self.maybe_empty_cache()
        return {"text_encoder_output": text_encoder_output}

    _run_input_encoder_local_t2av = _run_input_encoder_local_h3
    _run_input_encoder_local_i2av = _run_input_encoder_local_h3

    @ProfilingContext4DebugL2("Prepare DiT")
    def init_run(self):
        prompt_embeds = self.inputs["text_encoder_output"]["prompt_embeds"]
        self.scheduler.prepare(
            seed=self.input_info.seed,
            num_frames=self.request_num_frames,
            height=self.request_height,
            width=self.request_width,
            text_token_tags=self.inputs["text_encoder_output"]["text_token_tags"],
            keyframe_anchors=self.keyframe_anchors,
            condition_video_latents=self.condition_video_latents,
            condition_audio_latents=self.condition_audio_latents,
            references=self.prepared_references,
        )
        logger.info(
            "MiniMax-H3 packed layout: "
            f"text={prompt_embeds.shape[0]}, audio={self.scheduler.audio_latents.shape[0]}, "
            f"video={self.scheduler.video_latents.shape[0]}, total={self.scheduler.layout.sequence_length}"
        )
        if not self.config.get("cpu_offload", False):
            logger.info("MiniMax-H3 transformer is resident on the accelerator")
        elif self.config.get("offload_granularity", "model") == "model":
            logger.info("Moving the native MiniMax-H3 transformer to the accelerator")
            self.model.to_cuda()
        else:
            logger.info("MiniMax-H3 block offload enabled; keeping source blocks on CPU and using two accelerator buffers")
        torch_device_module.synchronize()

    def run_segment(self, segment_idx=0):
        infer_steps = self.scheduler.infer_steps
        for step_index in range(infer_steps):
            with ProfilingContext4DebugL1(
                "Run Dit every step",
                recorder_mode=GET_RECORDER_MODE(),
                metrics_func=monitor_cli.lightx2v_run_per_step_dit_duration,
                metrics_labels=[step_index + 1, infer_steps],
            ):
                self.check_stop()
                logger.info(f"==> MiniMax-H3 step: {step_index + 1} / {infer_steps}")
                with ProfilingContext4DebugL1("step_pre"):
                    self.scheduler.step_pre(step_index)
                with ProfilingContext4DebugL1("🚀 infer_main"):
                    self.model.infer(self.inputs)
                with ProfilingContext4DebugL1("step_post"):
                    self.scheduler.step_post()
                if self.progress_callback:
                    self.progress_callback(((step_index + 1) / infer_steps) * 100, 100)
        return self.scheduler.video_latents, self.scheduler.audio_latents

    @ProfilingContext4DebugL2("Offload DiT")
    def _offload_transformer(self):
        if not self.config.get("cpu_offload", False):
            return
        if self.config.get("offload_granularity", "model") == "block":
            logger.info("Offloading MiniMax-H3 pre/post weights; retaining the two block-offload device buffers")
            self.model.pre_weight.to_cpu()
            self.model.post_weight.to_cpu()
        else:
            logger.info("Offloading MiniMax-H3 transformer before VAE decode")
            self.model.to_cpu()
        torch_device_module.synchronize()
        self.maybe_empty_cache(force=True, collect_garbage=True)

    @ProfilingContext4DebugL1(
        "Run VAE Decoder",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_run_vae_decode_duration,
        metrics_labels=["MiniMaxH3Runner"],
    )
    def run_vae_decoder(self, video_rows, audio_rows):
        video_rows = video_rows[self.scheduler.num_condition_video_rows :]
        audio_rows = audio_rows[self.scheduler.num_condition_audio_rows :]
        video_latents = unpatchify_video_tokens(
            video_rows,
            self.scheduler.num_latent_frames,
            self.scheduler.latent_height,
            self.scheduler.latent_width,
            channels=int(self.config.get("in_channels", 24)),
            patch_size=tuple(self.config.get("patch_size", (1, 2, 2))),
        )
        audio_latents = unpack_audio_tokens(audio_rows, self.scheduler.num_audio_latents)
        with ProfilingContext4DebugL1("Run Video VAE Decoder"):
            video = self.video_vae.decode(video_latents)
        audio = None
        if not self.video_vae.decode_parallel or dist.get_rank() == 0:
            with ProfilingContext4DebugL1("Run Audio VAE Decoder"):
                audio = self.audio_vae.decode(audio_latents)
        return video, audio

    @staticmethod
    def _video_to_uint8_frames(video):
        if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 3:
            raise ValueError(f"decoded H3 video must be [1,3,F,H,W], got {tuple(video.shape)}")
        return (video[0].permute(1, 2, 3, 0).float() * 255.0).round().to(torch.uint8).contiguous().cpu()

    def process_images_after_vae_decoder(self):
        if self.video_vae.decode_parallel and dist.get_rank() != 0:
            return {"video": None, "audio": None}
        if self.input_info.return_result_tensor:
            return {
                # Match the public tensor layout of the reference pipeline:
                # [batch, frames, channels, height, width].
                "video": self.gen_video.permute(0, 2, 1, 3, 4).contiguous().cpu(),
                "audio": self.gen_audio.cpu(),
                "sampling_rate": self.audio_vae.sampling_rate,
            }

        output_path = self.input_info.save_result_path
        if output_path and (not dist.is_initialized() or dist.get_rank() == 0):
            from lightx2v.models.video_encoders.hf.ltx2.audio_vae.ops import Audio
            from lightx2v.utils.ltx2_media_io import encode_video

            if os.path.splitext(output_path)[1].lower() != ".mp4":
                raise ValueError(f"MiniMax-H3 AV output uses H.264/AAC; save_result_path must end in .mp4, got {output_path!r}")
            parent = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(parent, exist_ok=True)
            frames = self._video_to_uint8_frames(self.gen_video)
            waveform = self.gen_audio[0].float().cpu()
            audio = Audio(
                waveform=waveform,
                sampling_rate=self.audio_vae.sampling_rate,
            )
            logger.info(f"Saving MiniMax-H3 audio-video output to {output_path}")
            with ProfilingContext4DebugL2("Save Audio-Video Output"):
                encode_video(
                    video=frames,
                    fps=int(self.config.get("fps", 24)),
                    audio=audio,
                    output_path=output_path,
                    video_chunks_number=1,
                    video_codec_options=self.config.get("video_codec_options"),
                )
            logger.info(f"MiniMax-H3 output saved to {output_path}")
        return {"video": None, "audio": None}

    @ProfilingContext4DebugL2("Run DiT")
    def run_main(self):
        should_offload_transformer = self.config.get("cpu_offload", False)
        transformer_offloaded = not should_offload_transformer
        try:
            self.init_run()
            try:
                video_rows, audio_rows = self.run_segment(0)
            finally:
                if should_offload_transformer:
                    self._offload_transformer()
                    transformer_offloaded = True

            if self.config.get("h3_dit_only", False):
                output_path = self.config.get("h3_latent_output_path")
                if not output_path:
                    raise ValueError("MiniMax-H3 h3_dit_only requires h3_latent_output_path")
                video_finite = bool(torch.isfinite(video_rows).all())
                audio_finite = bool(torch.isfinite(audio_rows).all())
                if not video_finite or not audio_finite:
                    raise FloatingPointError(
                        f"MiniMax-H3 DiT-only gate produced non-finite latents: video={video_finite}, audio={audio_finite}"
                    )
                if not dist.is_initialized() or dist.get_rank() == 0:
                    output_path = os.path.abspath(output_path)
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    metadata = {
                        "video_shape": str(tuple(video_rows.shape)),
                        "audio_shape": str(tuple(audio_rows.shape)),
                        "video_min": f"{float(video_rows.min()):.9g}",
                        "video_max": f"{float(video_rows.max()):.9g}",
                        "audio_min": f"{float(audio_rows.min()):.9g}",
                        "audio_max": f"{float(audio_rows.max()):.9g}",
                        "finite": "true",
                        "model_evaluations": str(self.scheduler.infer_steps),
                    }
                    save_file(
                        {
                            "video_rows": video_rows.detach().float().cpu().contiguous(),
                            "audio_rows": audio_rows.detach().float().cpu().contiguous(),
                        },
                        output_path,
                        metadata=metadata,
                    )
                    logger.info(f"Saved MiniMax-H3 DiT-only finite gate to {output_path}: {metadata}")
                if dist.is_initialized():
                    dist.barrier()
                return {"h3_dit_only": output_path, "finite": True}

            self.gen_video, self.gen_audio = self.run_vae_decoder(video_rows, audio_rows)
            return self.process_images_after_vae_decoder()
        finally:
            # ``init_run`` can fail after a partial device transfer. Preserve
            # the original exception while still making a best-effort return
            # of the large transformer to host memory.
            if should_offload_transformer and not transformer_offloaded:
                with suppress(Exception):
                    self._offload_transformer()
            try:
                self.end_run()
            finally:
                # Decoded FP32 video is large (roughly 1.5 GiB at the default
                # shape). Returned tensors keep their own references/copies;
                # the runner should not retain another request-sized result.
                self.gen_video = None
                self.gen_audio = None
