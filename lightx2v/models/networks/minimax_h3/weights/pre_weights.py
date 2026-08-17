import torch.distributed as dist

from lightx2v.common.modules.weight_module import WeightModule, WeightModuleList
from lightx2v.models.networks.minimax_h3.weights.tensor_parallel import MiniMaxH3TensorParallelLinear
from lightx2v.utils.registry_factory import ATTN_WEIGHT_REGISTER, MM_WEIGHT_REGISTER, RMS_WEIGHT_REGISTER, TENSOR_REGISTER


def _linear(name, bias=False, force_fp32=False, config=None, tp_split=None):
    kind = "Default-ForceFp32" if force_fp32 else "Default"
    if config is not None and config.get("tensor_parallel", False) and tp_split is not None:
        tp_group = config["device_mesh"].get_group(mesh_dim="tensor_p")
        return MiniMaxH3TensorParallelLinear(
            weight_name=f"{name}.weight",
            bias_name=f"{name}.bias" if bias else None,
            mm_type=kind,
            tp_group=tp_group,
            tp_rank=dist.get_rank(tp_group),
            tp_size=dist.get_world_size(tp_group),
            split_dim=tp_split,
        )
    return MM_WEIGHT_REGISTER[kind](f"{name}.weight", f"{name}.bias" if bias else None)


def _rms(config, name, eps):
    return RMS_WEIGHT_REGISTER[config.get("rms_type", "torch_native")](name, eps=eps)


class MiniMaxH3RefinerAttentionWeights(WeightModule):
    def __init__(self, prefix, config, force_fp32=False):
        super().__init__()
        self.add_module("to_q", _linear(f"{prefix}.to_q", force_fp32=force_fp32, config=config, tp_split="col"))
        self.add_module("to_k", _linear(f"{prefix}.to_k", force_fp32=force_fp32, config=config, tp_split="col"))
        self.add_module("to_v", _linear(f"{prefix}.to_v", force_fp32=force_fp32, config=config, tp_split="col"))
        self.add_module(
            "norm_q",
            _rms(config, f"{prefix}.norm_q.weight", eps=float(config.get("qk_norm_eps", 1e-5))),
        )
        self.add_module(
            "norm_k",
            _rms(config, f"{prefix}.norm_k.weight", eps=float(config.get("qk_norm_eps", 1e-5))),
        )
        attn_type = config.get("attn_type", "flash_attn3")
        attention_cls = ATTN_WEIGHT_REGISTER[attn_type]
        if attn_type == "dynamic_sparse_attn":
            calculate = attention_cls(config.get("dynamic_sparse_attn_setting", {}))
        else:
            calculate = attention_cls()
        self.add_module("calculate", calculate)
        self.add_module("to_out", _linear(f"{prefix}.to_out.0", force_fp32=force_fp32, config=config, tp_split="row"))


class MiniMaxH3FeedForwardWeights(WeightModule):
    def __init__(self, prefix, config, force_fp32=False):
        super().__init__()
        self.add_module("in_proj", _linear(f"{prefix}.net.0.proj", force_fp32=force_fp32, config=config, tp_split="col"))
        self.add_module("out_proj", _linear(f"{prefix}.net.2", force_fp32=force_fp32, config=config, tp_split="row"))


class MiniMaxH3TokenRefinerBlockWeights(WeightModule):
    def __init__(self, index, config):
        super().__init__()
        prefix = f"token_refiner.refiner_blocks.{index}"
        eps = float(config.get("norm_eps", 1e-5))
        force_fp32 = bool(config.get("h3_v100_fp16", False))
        self.add_module("norm1", _rms(config, f"{prefix}.norm1.weight", eps=eps))
        self.add_module("attn", MiniMaxH3RefinerAttentionWeights(f"{prefix}.attn", config, force_fp32=force_fp32))
        self.add_module("norm2", _rms(config, f"{prefix}.norm2.weight", eps=eps))
        self.add_module("ff", MiniMaxH3FeedForwardWeights(f"{prefix}.ff", config, force_fp32=force_fp32))


class MiniMaxH3PreWeights(WeightModule):
    def __init__(self, config):
        super().__init__()
        # The full checkpoint keeps the timestep MLP in fp32. Curve-form
        # checkpoints replace that MLP with an fp32 interpolation table.
        self.add_module("proj_in", _linear("proj_in", bias=True, force_fp32=True))
        self.add_module("audio_proj_in", _linear("audio_proj_in", bias=True, force_fp32=True))
        self.add_module("context_embedder", _linear("context_embedder", bias=True, force_fp32=bool(config.get("h3_v100_fp16", False))))
        if config.get("h3_adaln_curve", False):
            self.register_parameter("adaln_t_table", TENSOR_REGISTER["Default"]("adaln_t_table"))
        else:
            self.add_module("time_linear_1", _linear("time_embedder.linear_1", bias=True, force_fp32=True))
            self.add_module("time_linear_2", _linear("time_embedder.linear_2", bias=True, force_fp32=True))
        self.add_module(
            "refiner_blocks",
            WeightModuleList([MiniMaxH3TokenRefinerBlockWeights(i, config) for i in range(int(config.get("num_refiner_layers", 2)))]),
        )
        self.add_module(
            "refiner_final_norm",
            _rms(config, "token_refiner.final_norm.weight", eps=float(config.get("final_norm_eps", 1e-5))),
        )
