import json
import os

import torch
from safetensors.torch import save_file


class EncodeMiniMaxH3TextConditioning:
    """Encode pure T2AV text without loading a DiT or VAE."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "encode"
    CATEGORY = "MiniMax/H3 diagnostics"

    def encode(self, clip, prompt):
        tokens = clip.tokenize(prompt)
        return (clip.encode_from_tokens_scheduled(tokens),)


class SaveMiniMaxH3Conditioning:
    """Save text plus optional keyframe/reference latents used by LightX2V."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "output_path": ("STRING", {"default": "/tmp/minimax-h3-conditioning.safetensors"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax/H3 diagnostics"

    def save(self, conditioning, output_path):
        if len(conditioning) != 1:
            raise ValueError(f"Expected one H3 conditioning entry, got {len(conditioning)}")
        prompt_embeds, metadata = conditioning[0]
        token_tags = metadata.get("minimax_token_tags")
        if token_tags is None:
            raise ValueError("H3 conditioning is missing minimax_token_tags")
        if prompt_embeds.ndim == 3 and prompt_embeds.shape[0] == 1:
            prompt_embeds = prompt_embeds.squeeze(0)
        if token_tags.ndim == 2 and token_tags.shape[0] == 1:
            token_tags = token_tags.squeeze(0)
        if prompt_embeds.ndim != 2 or prompt_embeds.shape[-1] != 5120:
            raise ValueError(f"Expected prompt embeddings [tokens, 5120], got {tuple(prompt_embeds.shape)}")
        if token_tags.ndim != 1 or token_tags.shape[0] != prompt_embeds.shape[0]:
            raise ValueError(
                f"Expected one token tag per embedding row, got embeds={tuple(prompt_embeds.shape)}, "
                f"tags={tuple(token_tags.shape)}"
            )
        tensors = {
            "prompt_embeds": prompt_embeds.detach().cpu().contiguous(),
            "text_token_tags": token_tags.detach().to(torch.int64).cpu().contiguous(),
        }
        bundle = {
            "format": "minimax_h3_conditioning_bundle",
            "version": 1,
            "task": "t2av",
            "keyframes": [],
            "references": [],
        }

        keyframes = metadata.get("minimax_keyframes") or []
        references = metadata.get("minimax_refs") or []
        if keyframes and references:
            raise ValueError("H3 conditioning cannot contain keyframes and references together")

        if keyframes:
            frame_count = int(metadata.get("minimax_frame_count", 0))
            if frame_count <= 0:
                raise ValueError("H3 keyframe conditioning is missing minimax_frame_count")
            anchors = []
            for index, keyframe in enumerate(keyframes):
                latent = keyframe.get("latent")
                if not isinstance(latent, torch.Tensor) or latent.ndim != 5:
                    raise ValueError(
                        f"Expected keyframe {index} latent [B,C,T,H,W], got {type(latent).__name__} "
                        f"{getattr(latent, 'shape', None)}"
                    )
                resolved_index = int(keyframe["resolved_frame_index"])
                if resolved_index == 0:
                    anchor = "first"
                elif resolved_index == frame_count - 1:
                    anchor = "last"
                else:
                    raise ValueError(
                        f"H3 bundle only supports first/last keyframes, got frame {resolved_index} of {frame_count}"
                    )
                tensor_name = f"keyframe_{index}_latent"
                tensors[tensor_name] = latent.detach().cpu().contiguous()
                anchors.append(anchor)
                bundle["keyframes"].append(
                    {
                        "anchor": anchor,
                        "resolved_frame_index": resolved_index,
                        "tensor": tensor_name,
                        "shape": list(latent.shape),
                    }
                )
            bundle["frame_count"] = frame_count
            bundle["task"] = {("first",): "i2av", ("last",): "l2av", ("first", "last"): "fl2av"}.get(
                tuple(anchors)
            )
            if bundle["task"] is None:
                raise ValueError(f"Unsupported H3 keyframe anchor sequence: {anchors}")

        if references:
            bundle["task"] = "ref2av"
            for index, reference in enumerate(references):
                source_kind = str(reference.get("kind", ""))
                kind = "video" if source_kind == "video_audio" else source_kind
                if kind not in {"image", "video", "audio"}:
                    raise ValueError(f"Unsupported H3 reference kind at index {index}: {source_kind!r}")
                entry = {
                    "kind": kind,
                    "has_audio": bool(reference.get("audio_latent") is not None),
                }
                latent = reference.get("latent")
                if kind != "audio":
                    if not isinstance(latent, torch.Tensor) or latent.ndim != 5:
                        raise ValueError(
                            f"Expected reference {index} video latent [B,C,T,H,W], got {type(latent).__name__} "
                            f"{getattr(latent, 'shape', None)}"
                        )
                    tensor_name = f"reference_{index}_video_latent"
                    tensors[tensor_name] = latent.detach().cpu().contiguous()
                    entry.update(
                        {
                            "video_tensor": tensor_name,
                            "num_latent_frames": int(reference.get("latent_t", latent.shape[2])),
                            "latent_height": int(reference.get("latent_h", latent.shape[3])),
                            "latent_width": int(reference.get("latent_w", latent.shape[4])),
                            "video_shape": list(latent.shape),
                        }
                    )
                audio_latent = reference.get("audio_latent")
                if audio_latent is not None:
                    if not isinstance(audio_latent, torch.Tensor):
                        raise ValueError(f"Expected reference {index} audio latent tensor")
                    tensor_name = f"reference_{index}_audio_latent"
                    tensors[tensor_name] = audio_latent.detach().cpu().contiguous()
                    entry.update(
                        {
                            "audio_tensor": tensor_name,
                            "num_audio_latents": int(reference.get("ref_audio_t", audio_latent.shape[-1])),
                            "audio_shape": list(audio_latent.shape),
                        }
                    )
                bundle["references"].append(entry)

        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_file(
            tensors,
            output_path,
            metadata={
                "source": "MiniMax-H3 conditioning export for LightX2V",
                "prompt_shape": str(tuple(prompt_embeds.shape)),
                "prompt_dtype": str(prompt_embeds.dtype),
                "minimax_h3_bundle": json.dumps(bundle, ensure_ascii=False, separators=(",", ":")),
            },
        )
        return {
            "ui": {
                "text": [
                    f"saved {bundle['task']} {tuple(prompt_embeds.shape)} with "
                    f"{len(bundle['keyframes'])} keyframe(s) and {len(bundle['references'])} reference(s) "
                    f"to {output_path}"
                ]
            }
        }


NODE_CLASS_MAPPINGS = {
    "EncodeMiniMaxH3TextConditioning": EncodeMiniMaxH3TextConditioning,
    "SaveMiniMaxH3Conditioning": SaveMiniMaxH3Conditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "EncodeMiniMaxH3TextConditioning": "Encode MiniMax-H3 Text Conditioning",
    "SaveMiniMaxH3Conditioning": "Save MiniMax-H3 Conditioning",
}
