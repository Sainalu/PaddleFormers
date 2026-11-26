"""Strict original-weight loading and explicit multimodal compilation boundary."""

import hashlib
from copy import deepcopy
import json
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

from common import ROOT, sha256


sys.path.insert(0, os.environ.get("PADDLEFORMERS_DIR", str(ROOT.parents[3])))


def checkpoint_header(path):
    with Path(path).open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 2 <= length <= 16 * 1024 * 1024:
            raise ValueError("invalid safetensors header")
        return json.loads(stream.read(length)), 8 + length


def mapping(config):
    from paddleformers.transformers import AyaVisionForConditionalGeneration

    statements = AyaVisionForConditionalGeneration._gen_aoa_config(config)["aoa_statements"]
    rules = []
    for statement in statements:
        match = re.fullmatch(r"([\w.]+)(\^T)? -> ([\w.]+)(?:,.*)?", statement)
        if match is None:
            raise ValueError(f"unsupported conversion statement: {statement}")
        rules.append({"source": match.group(1), "target": match.group(3),
                      "transpose": bool(match.group(2))})
    return rules


def load_model(checkpoint, source_metadata=None):
    import paddle
    from paddleformers.transformers import AyaVisionConfig, AyaVisionForConditionalGeneration

    checkpoint = Path(checkpoint)
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, entry in manifest["files"].items():
        if sha256(checkpoint / name) != entry["sha256"]:
            raise ValueError(f"checkpoint file hash mismatch: {name}")
    if source_metadata is None:
        source_metadata = Path(os.environ["AYA_VISION_METADATA_DIR"])
    source_metadata = Path(source_metadata)
    source_path = source_metadata / "config.json"
    source_index = source_metadata / "model.safetensors.index.json"
    if sha256(source_path) != manifest["source_config_sha256"]:
        raise ValueError("original source config hash mismatch")
    if sha256(source_index) != manifest["source_index_sha256"]:
        raise ValueError("original source index hash mismatch")
    source_config = json.loads(source_path.read_text())
    original = json.loads((checkpoint / "config.json").read_text())
    expected = deepcopy(source_config)
    expected["text_config"]["num_hidden_layers"] = 4
    expected["vision_config"]["num_hidden_layers"] = 2
    if "layer_types" in expected["text_config"]:
        expected["text_config"]["layer_types"] = expected["text_config"]["layer_types"][:4]
    if original != expected:
        raise ValueError("reduced config changed original dimensions, scaling or other non-layer settings")
    if source_config["text_config"]["num_hidden_layers"] != 32 or source_config["vision_config"]["num_hidden_layers"] != 27:
        raise ValueError("unexpected original source layer counts")
    if original["model_type"] != "aya_vision":
        raise ValueError("expected actual Aya Vision checkpoint")
    config = AyaVisionConfig(**original)
    for subconfig in (config, config.text_config, config.vision_config):
        subconfig._attn_implementation = "eager"
        subconfig.use_cache = False
        subconfig.dtype = "float32"
        subconfig.tensor_parallel_degree = 1
        subconfig.tensor_model_parallel_size = 1
        subconfig.use_fused_head_and_loss_fn = False
        subconfig.use_filtered_label_loss = False
    if set(config.text_config.layer_types) != {"sliding_attention", "full_attention"}:
        raise ValueError("retained text prefix must cover both attention types")
    if config.text_config.num_hidden_layers != 4 or config.vision_config.num_hidden_layers != 2:
        raise ValueError("expected four original text layers and two original vision layers")
    if not config.tie_word_embeddings:
        raise ValueError("review untied actual checkpoint head mapping before loading")
    if config.vision_config.vision_use_head:
        raise ValueError("review pooled vision head mapping for this actual checkpoint")
    paddle.set_default_dtype("float32")
    model = AyaVisionForConditionalGeneration(config)
    model.tie_weights()
    if model.lm_head.weight is not model.model.language_model.embed_tokens.weight:
        raise ValueError("output head was not tied to original embedding")
    parameters = model.state_dict()
    rules = mapping(config)
    targets = {rule["target"] for rule in rules}
    if len(targets) != len(rules) or targets != set(parameters):
        raise ValueError(f"Paddle mapping incomplete: missing={sorted(set(parameters) - targets)} "
                         f"unexpected={sorted(targets - set(parameters))}")
    weights = checkpoint / "model.safetensors"
    header, data_offset = checkpoint_header(weights)
    sources = set(header) - {"__metadata__"}
    needed = {rule["source"] for rule in rules}
    if needed != sources:
        raise ValueError(f"HF mapping incomplete: missing={sorted(needed - sources)} "
                         f"unexpected={sorted(sources - needed)}")
    if not manifest.get("retained_tensor_bytes_verified"):
        raise ValueError("original byte-range provenance is required")
    if set(manifest["tensors"]) != sources:
        raise ValueError("manifest does not account for every original tensor")
    loaded = {}
    records = []
    by_source = {}
    for rule in rules:
        by_source.setdefault(rule["source"], []).append(rule)
    with weights.open("rb") as stream:
        for source_name, source_rules in sorted(by_source.items()):
            entry = header[source_name]
            first, last = entry["data_offsets"]
            stream.seek(data_offset + first)
            raw = stream.read(last - first)
            digest = hashlib.sha256(raw).hexdigest()
            provenance = manifest["tensors"][source_name]
            if digest != provenance["sha256"] or entry["shape"] != provenance["shape"]:
                raise ValueError(f"original tensor hash/shape mismatch: {source_name}")
            if entry["dtype"] == "BF16":
                array = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
            elif entry["dtype"] in ("F16", "F32", "F64"):
                numpy_dtype = {"F16": "<f2", "F32": "<f4", "F64": "<f8"}[entry["dtype"]]
                array = np.frombuffer(raw, dtype=numpy_dtype).astype(np.float32)
            else:
                raise ValueError(f"unsupported original tensor dtype: {source_name}")
            array = array.reshape(entry["shape"])
            if not np.isfinite(array).all():
                raise ValueError(f"nonfinite original tensor: {source_name}")
            tensor = None
            for rule in source_rules:
                converted = np.ascontiguousarray(array.T if rule["transpose"] else array)
                parameter = parameters[rule["target"]]
                if list(converted.shape) != list(parameter.shape):
                    raise ValueError(f"tensor shape mismatch: {source_name} -> {rule['target']}")
                if tensor is None or rule["transpose"]:
                    tensor = paddle.to_tensor(converted, dtype="float32")
                loaded[rule["target"]] = tensor
                records.append({**rule, "source_shape": entry["shape"],
                                "paddle_shape": list(parameter.shape), "source_dtype": entry["dtype"],
                                "compute_dtype": "float32", "original_sha256": digest,
                                "loaded_fp32_sha256": hashlib.sha256(converted).hexdigest()})
    missing, unexpected = model.set_state_dict(loaded)
    if missing or unexpected:
        raise ValueError(f"strict Paddle load failed: {missing}, {unexpected}")
    if model.lm_head.weight is not model.model.language_model.embed_tokens.weight:
        raise ValueError("strict loading broke head sharing")
    if {str(parameter.dtype) for parameter in model.parameters()} != {"paddle.float32"}:
        raise ValueError("all parameters must compute in FP32")
    buffers_before = {name: {"place": str(buffer.place), "dtype": str(buffer.dtype)}
                      for name, buffer in model.named_buffers()}
    model.to(device=paddle.get_device())
    expected_place = str(next(iter(model.parameters())).place)
    buffers = {}
    for name, buffer in model.named_buffers():
        if str(buffer.place) != expected_place or str(buffer.dtype) != buffers_before[name]["dtype"]:
            raise ValueError(f"buffer device migration changed dtype or left a buffer on another device: {name}")
        if "position_ids" in name and buffer.dtype != paddle.int64:
            raise ValueError("SigLIP position IDs must remain int64")
        if "inv_freq" in name and buffer.dtype != paddle.float32:
            raise ValueError("Cohere2 RoPE inverse frequency must remain float32")
        buffers[name] = {"before": buffers_before[name], "place": str(buffer.place),
                         "dtype": str(buffer.dtype), "shape": list(buffer.shape)}
    if model.lm_head.weight is not model.model.language_model.embed_tokens.weight:
        raise ValueError("device migration broke tied output embedding")
    report = {"strict_loading": True, "original_tensors": len(sources), "mapped_targets": len(records),
              "all_source_tensor_hashes_verified": True, "head_shared": True,
              "checkpoint_sha256": manifest["files"][weights.name]["sha256"],
              "checkpoint_manifest_sha256": sha256(manifest_path),
              "config_sha256": sha256(checkpoint / "config.json"),
              "source_repo": manifest["source_repo"], "source_revision": manifest["source_revision"],
              "attention_types": config.text_config.layer_types, "tensor_mapping": records,
              "source_config_sha256": sha256(source_path), "source_index_sha256": sha256(source_index),
              "all_checkpoint_file_hashes_verified": True,
              "buffers": buffers, "all_buffer_places_verified": True,
              "source_config_unchanged_except_prefix_layer_counts": True,
              "original_source_dims": {"text": source_config["text_config"], "vision": source_config["vision_config"],
                                       "downsample_factor": source_config["downsample_factor"],
                                       "alignment_intermediate_size": source_config["alignment_intermediate_size"],
                                       "vision_feature_layer": source_config["vision_feature_layer"]},
              "parameter_count": sum(int(np.prod(parameter.shape)) for parameter in model.parameters())}
    return model, report


def graph_class():
    import paddle

    class MultimodalGraph(paddle.nn.Layer):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.train() if model.training else self.eval()

        def forward(self, input_ids, pixel_values, attention_mask, position_ids):
            outputs = self.model(input_ids, pixel_values, attention_mask, position_ids,
                                 use_cache=False, return_dict=False)
            return outputs[0], outputs[4]

    return MultimodalGraph


def loss_class():
    import paddle

    class MultimodalLoss(paddle.nn.Layer):
        def __init__(self, model):
            super().__init__()
            self.graph = graph_class()(model)
            self.train() if model.training else self.eval()

        def forward(self, input_ids, pixel_values, attention_mask, position_ids, labels):
            logits, image_features = self.graph(input_ids, pixel_values, attention_mask, position_ids)
            shifted_labels = labels[:, 1:]
            loss = paddle.nn.functional.cross_entropy(logits[:, :-1, :], shifted_labels,
                                                       ignore_index=-100, reduction="sum")
            count = paddle.sum((shifted_labels != -100).astype("float32"))
            return loss / count

    return MultimodalLoss
