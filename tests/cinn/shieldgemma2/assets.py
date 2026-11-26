"""Audit real ShieldGemma2 assets and rebuild fixed inputs on a locked CPU source."""

import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np


PROJECT = Path(__file__).resolve().parents[3]
MODEL = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT
DEFAULT_CHECKPOINT = MODEL / "checkpoints/shieldgemma2-2text-2vision-bf16"
DEFAULT_DATA = MODEL / "data/docling-structure-20261018"
DEFAULT_IMAGES = PROJECT / "shared_data/multimodal_sft_docling"
SOURCE_LOCK = Path(__file__).with_name("source-lock.json")
INPUT_NAMES = ("input_ids", "pixel_values", "attention_mask", "position_ids", "token_type_ids")
CHANGED_FILES = {
    "paddleformers/transformers/gemma3/modeling.py",
    "paddleformers/transformers/gemma3/multimodal_text_modeling.py",
    "paddleformers/transformers/shieldgemma2/modeling.py",
}
CHECKPOINT_FILES = {"added_tokens.json", "chat_template.json", "config.json", "model.safetensors",
                    "preprocessor_config.json", "processor_config.json", "special_tokens_map.json",
                    "tokenizer.json", "tokenizer.model", "tokenizer_config.json"}
CHECKPOINT_MANIFEST_SHA256 = "f6c233b7fccc54e634be3ca8b2173f69143841a3ae052c5541c7237072ea81c6"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(), object_pairs_hook=unique)


def inside(root, name):
    relative = Path(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "asset path escapes its root")
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), "asset symlink escapes its root")
    return path


def reference(path):
    path = Path(path).resolve()
    require(path.is_relative_to(PROJECT), "reported assets must be inside the CINN project")
    return {"path": path.relative_to(PROJECT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def source_snapshot(source=DEFAULT_SOURCE, lock_path=SOURCE_LOCK):
    source = Path(source).resolve()
    lock_path = Path(lock_path).resolve()
    lock = read_json(lock_path)
    require(set(lock["changed_files"]) == CHANGED_FILES, "source lock must contain exactly the three Shield/Gemma files")
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(source), *arguments], timeout=30)
    commit = git("rev-parse", "HEAD").decode().strip()
    branch = git("branch", "--show-current").decode().strip()
    difference = git("diff", "--binary", "HEAD")
    changed = git("diff", "--name-only", "HEAD").decode().splitlines()
    patch = inside(MODEL, lock["patch"])
    runner_root = MODEL.parent.resolve()
    untracked = git("ls-files", "--others", "--exclude-standard").decode().splitlines()
    outside_runner = []
    for name in untracked:
        path = (source / name).resolve()
        if path.suffix in (".py", ".so") and not path.is_relative_to(runner_root):
            outside_runner.append(name)
    require(commit == lock["commit"], "source must remain at the source-lock commit")
    require(set(changed) == CHANGED_FILES and len(changed) == 3, "source has unrelated or missing changes")
    require(sha256(patch) == lock["patch_sha256"] and difference == patch.read_bytes(), "source patch differs from its lock")
    require(not git("diff", "--cached") and not outside_runner,
            "source contains staged or untracked files outside the locked patch")
    return {"source_commit": commit, "branch": branch, "upstream_head": commit, "detached": not branch,
            "source_diff_sha256": hashlib.sha256(difference).hexdigest(),
            "diff_sha256": hashlib.sha256(difference).hexdigest(), "changed_files": changed,
            "patch_sha256": sha256(patch), "patch": reference(patch), "source_lock": reference(lock_path),
            "source_lock_sha256": sha256(lock_path), "path": source.relative_to(PROJECT).as_posix()}


def verify_manifest_files(directory, manifest):
    require(set(manifest["files"]) == CHECKPOINT_FILES, "checkpoint manifest must account for exactly ten files")
    records = {}
    for name, expected in manifest["files"].items():
        path = inside(directory, name)
        require(path.is_file() and path.stat().st_size == expected["bytes"], f"checkpoint file size mismatch: {name}")
        digest = sha256(path)
        require(digest == expected["sha256"], f"checkpoint file hash mismatch: {name}")
        records[name] = {"bytes": path.stat().st_size, "sha256": digest}
    return records


def expected_shapes():
    shapes = {
        "model.language_model.model.embed_tokens.weight": [262208, 2560],
        "model.language_model.model.norm.weight": [2560],
        "model.multi_modal_projector.mm_input_projection_weight": [1152, 2560],
        "model.multi_modal_projector.mm_soft_emb_norm.weight": [1152],
        "model.vision_tower.vision_model.embeddings.patch_embedding.bias": [1152],
        "model.vision_tower.vision_model.embeddings.patch_embedding.weight": [1152, 3, 14, 14],
        "model.vision_tower.vision_model.embeddings.position_embedding.weight": [4096, 1152],
        "model.vision_tower.vision_model.post_layernorm.bias": [1152],
        "model.vision_tower.vision_model.post_layernorm.weight": [1152],
    }
    text = {"mlp.down_proj.weight": [2560, 10240], "mlp.gate_proj.weight": [10240, 2560],
            "mlp.up_proj.weight": [10240, 2560], "self_attn.k_proj.weight": [1024, 2560],
            "self_attn.v_proj.weight": [1024, 2560], "self_attn.q_proj.weight": [2048, 2560],
            "self_attn.o_proj.weight": [2560, 2048], "self_attn.k_norm.weight": [256],
            "self_attn.q_norm.weight": [256]}
    for name in ("input_layernorm", "post_attention_layernorm", "post_feedforward_layernorm", "pre_feedforward_layernorm"):
        text[name + ".weight"] = [2560]
    vision = {"mlp.fc1.bias": [4304], "mlp.fc1.weight": [4304, 1152],
              "mlp.fc2.bias": [1152], "mlp.fc2.weight": [1152, 4304]}
    for name in ("layer_norm1", "layer_norm2", "self_attn.k_proj", "self_attn.q_proj", "self_attn.v_proj", "self_attn.out_proj"):
        vision[name + ".bias"] = [1152]
        vision[name + ".weight"] = [1152] if name.startswith("layer_norm") else [1152, 1152]
    for layer in range(2):
        shapes.update({f"model.language_model.model.layers.{layer}.{name}": shape for name, shape in text.items()})
        shapes.update({f"model.vision_tower.vision_model.encoder.layers.{layer}.{name}": shape for name, shape in vision.items()})
    return shapes


def validate_header(header, payload_bytes):
    entries = {name: value for name, value in header.items() if name != "__metadata__"}
    shapes = expected_shapes()
    require(len(entries) == 67 and set(entries) == set(shapes), "expected exactly 67 original prefix tensor names")
    intervals = []
    for name, entry in entries.items():
        require(entry["dtype"] == "BF16" and entry["shape"] == shapes[name], f"BF16 tensor geometry mismatch: {name}")
        offsets = entry["data_offsets"]
        require(len(offsets) == 2 and all(type(value) is int and value >= 0 for value in offsets), "invalid tensor offsets")
        require(offsets[1] - offsets[0] == math.prod(entry["shape"]) * 2, f"BF16 tensor byte count mismatch: {name}")
        intervals.append((offsets[0], offsets[1]))
    cursor = 0
    for first, last in sorted(intervals):
        require(first == cursor, "tensor offsets overlap or leave unaccounted bytes")
        cursor = last
    require(cursor == payload_bytes, "safetensors data boundary differs from tensor payload")
    return {"tensor_count": 67, "dtype": "BF16", "all_tensor_names_shapes_and_boundaries_verified": True,
            "tensor_shapes": shapes, "tensor_payload_bytes": cursor}


def audit_checkpoint(checkpoint=DEFAULT_CHECKPOINT):
    checkpoint = Path(checkpoint).resolve()
    manifest_path = checkpoint / "checkpoint_manifest.json"
    require(sha256(manifest_path) == CHECKPOINT_MANIFEST_SHA256, "original checkpoint manifest hash changed")
    manifest = read_json(manifest_path)
    files = verify_manifest_files(checkpoint, manifest)
    require(manifest["retained_layers"] == {"text": 2, "vision": 2} and manifest["tensor_count"] == 67,
            "original reduced layer/tensor accounting changed")
    config = read_json(checkpoint / "config.json")
    original_root = Path(manifest["source"])
    original_path = original_root / "config.json"
    index_path = original_root / "model.safetensors.index.json"
    require(sha256(original_path) == manifest["source_sha256"]["config.json"]
            and sha256(index_path) == manifest["source_sha256"][index_path.name], "original config/index hash differs")
    original = read_json(original_path)
    reduced = deepcopy(original)
    reduced["text_config"]["num_hidden_layers"] = 2
    reduced["vision_config"]["num_hidden_layers"] = 2
    require(config == reduced, "reduction changed non-layer dimensions, scaling or configuration")
    require(config["vision_config"]["image_size"] == 896 and config["mm_tokens_per_image"] == 256
            and config["text_config"]["hidden_size"] == 2560 and config["text_config"]["vocab_size"] == 262208,
            "original image/token/hidden/vocab geometry changed")
    weight_path = checkpoint / "model.safetensors"
    with weight_path.open("rb") as stream:
        header_bytes = struct.unpack("<Q", stream.read(8))[0]
        require(2 <= header_bytes <= 1024 * 1024, "invalid safetensors header size")
        header = json.loads(stream.read(header_bytes))
    geometry = validate_header(header, weight_path.stat().st_size - 8 - header_bytes)
    index = read_json(index_path)["weight_map"]
    selected = set()
    for name in index:
        retained = True
        for prefix in ("model.language_model.model.layers.", "model.vision_tower.vision_model.encoder.layers."):
            if name.startswith(prefix):
                retained = int(name[len(prefix):].split(".", 1)[0]) < 2
        if retained:
            selected.add(name)
    require(selected == set(geometry["tensor_shapes"]), "retained names differ from original index prefix")
    return {"passed": True, "manifest": reference(manifest_path), "files": files,
            "directory": checkpoint.relative_to(PROJECT).as_posix(), "all_ten_file_hashes_verified": True,
            "geometry": geometry, "config": config, "original_config_sha256": sha256(original_path),
            "original_index_sha256": sha256(index_path), "full_original_shards_rehashed": False,
            "original_config_unchanged_except_layer_counts": True,
            "original_tensor_claim_scope": "retained checkpoint file hash and original prefix names verified; original shard bytes not reread"}


def validate_provenance(record, entry):
    provenance = record.get("label_provenance")
    require(type(record.get("violates")) is bool and record["violates"] is False and entry["label"] == 1,
            "existing negative-only structural label changed")
    require(isinstance(provenance, dict) and provenance == entry.get("label_provenance"), "label provenance missing or inconsistent")
    require(provenance.get("independent_human_validation") is False and provenance.get("observation")
            and "AI coding assistant" in provenance.get("annotator", "")
            and "not original dataset supervision" in provenance.get("method", "")
            and "structure regression" in provenance.get("scope", ""), "labels cannot be represented as human gold safety data")


def validate_batch(batch):
    require(set(batch) == set(INPUT_NAMES) | {"labels"}, "NPZ must contain exactly five inputs and original labels")
    for name in ("input_ids", "attention_mask", "position_ids", "token_type_ids"):
        require(batch[name].dtype == np.int64 and batch[name].shape == (1, 448), f"unexpected {name} shape/dtype")
    ids, mask = batch["input_ids"], batch["attention_mask"]
    require(np.all((ids >= 0) & (ids < 262208)), "input token outside original vocabulary")
    require(np.isin(mask, [0, 1]).all() and mask[0, -1] == 1 and np.all(np.diff(mask, axis=1) >= 0), "expected contiguous left padding")
    positions = np.maximum(np.cumsum(mask, axis=-1) - 1, 0)
    require(np.array_equal(batch["position_ids"], positions), "position_ids differ from exact unpadded positions")
    image_mask = ids == 262144
    require(int(image_mask.sum()) == 256 and np.all(mask[image_mask] == 1), "expected 256 valid image placeholders")
    require(np.array_equal(batch["token_type_ids"], image_mask.astype(np.int64)), "image token_type_ids mismatch")
    pixels = batch["pixel_values"]
    require(pixels.shape == (1, 3, 896, 896) and pixels.dtype == np.float32 and np.isfinite(pixels).all(), "invalid original pixel geometry/dtype")
    require(batch["labels"].dtype == np.int64 and batch["labels"].shape == (1,) and batch["labels"][0] == 1,
            "NPZ labels differ from existing negative-only structural regression")
    return {name: {"shape": list(array.shape), "dtype": str(array.dtype),
                   "sha256": hashlib.sha256(array.tobytes()).hexdigest()} for name, array in batch.items()}


def assert_disjoint(train, evaluation):
    for key in ("image", "image_sha256", "batch_sha256"):
        first, second = [entry[key] for entry in train], [entry[key] for entry in evaluation]
        require(len(set(first)) == len(first) and len(set(second)) == len(second)
                and not set(first).intersection(second), f"train/eval leakage or duplicates by {key}")


def audit_data(checkpoint=DEFAULT_CHECKPOINT, dataset=DEFAULT_DATA, image_root=DEFAULT_IMAGES):
    checkpoint, dataset, image_root = map(lambda path: Path(path).resolve(), (checkpoint, dataset, image_root))
    dataset_manifest = read_json(dataset / "manifest.json")
    require(dataset_manifest["scope"] == "ai_reviewed_negative_only_structure_regression"
            and dataset_manifest["formal_acceptance"] is False, "dataset scope was changed to safety acceptance")
    require(sha256(image_root / "manifest.json") == dataset_manifest["source_manifest_sha256"], "shared Docling manifest hash differs")
    for name, digest in dataset_manifest["files_sha256"].items():
        require(sha256(inside(dataset, name)) == digest, f"dataset file hash differs: {name}")
    reports, loaded = {}, {}
    source_files = {}
    for split, expected_source in (("train", "mini_train"), ("eval", "eval")):
        directory = dataset / f"{split}-batches"
        manifest = read_json(directory / "manifest.json")
        records = [json.loads(line) for line in (dataset / f"{split}.jsonl").read_text().splitlines()]
        require(manifest["structure_only"] is True and manifest["max_length"] == 448
                and manifest["label_order"] == ["Yes/violates", "No/complies"], "batch scope/shape/label ordering differs")
        require(manifest["input_sha256"] == sha256(dataset / f"{split}.jsonl")
                and manifest["dataset_card_sha256"] == sha256(dataset / "DATASET_CARD.md")
                and manifest["checkpoint_config_sha256"] == sha256(checkpoint / "config.json"), "batch metadata hash differs")
        require(len(records) == len(manifest["batches"]) == 4, "expected four real images per split")
        entries, batches = [], []
        for record, entry in zip(records, manifest["batches"]):
            require(record["source_split"] == expected_source and record["policy"] == entry["policy"] == "violence"
                    and record["image"] == entry["image"], "original image/policy/source selection differs")
            validate_provenance(record, entry)
            source_path = image_root / f"{expected_source}.jsonl"
            if expected_source not in source_files:
                source_files[expected_source] = [json.loads(line) for line in source_path.read_text().splitlines()]
            require(record["source_jsonl_sha256"] == sha256(source_path), "original source text hash differs")
            require(source_files[expected_source][record["source_row_zero_based"]]["images"] == [record["image"]], "original Docling source row/image mismatch")
            image = inside(image_root, record["image"])
            require(sha256(image) == record["image_sha256"] == entry["image_sha256"], "original image hash differs")
            path = inside(directory, entry["file"])
            require(sha256(path) == entry["sha256"], "fixed NPZ hash differs")
            with np.load(path, allow_pickle=False) as archive:
                batch = {name: archive[name].copy() for name in archive.files}
            arrays = validate_batch(batch)
            entries.append({"image": reference(image)["path"], "image_sha256": sha256(image),
                            "batch_sha256": sha256(path), "batch": reference(path), "arrays": arrays,
                            "label": 1, "label_provenance": record["label_provenance"],
                            "policy": "violence", "source": reference(source_path),
                            "source_row_zero_based": record["source_row_zero_based"]})
            batches.append(batch)
        reports[split] = {"manifest": reference(directory / "manifest.json"), "records": reference(dataset / f"{split}.jsonl"),
                          "entries": entries}
        loaded[split] = batches
    assert_disjoint(reports["train"]["entries"], reports["eval"]["entries"])
    return {"passed": True, "train_eval_disjoint": True, "image_count": 8, "batch_count": 8,
            "dataset_manifest": reference(dataset / "manifest.json"), "dataset_card": reference(dataset / "DATASET_CARD.md"),
            "shared_manifest": reference(image_root / "manifest.json"), "splits": reports,
            "label_provenance_verified": True, "gold_safety_labels": False, "safety_accuracy_gate": False,
            "scope": "existing AI-reviewed negative-only structural regression; not safety capability evaluation"}, loaded


def clean_imports(source):
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    import paddle
    import paddleformers
    require(Path(paddleformers.__file__).resolve().is_relative_to(source), "PaddleFormers was imported from another source tree")
    paddle.set_device("cpu")
    paddle.set_default_dtype("float32")
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.set_num_threads(8)
        if torch.get_num_interop_threads() != 8:
            torch.set_num_interop_threads(8)
    return paddle


def rebuild_processor(source, checkpoint, data_report, loaded):
    clean_imports(source)
    from PIL import Image
    from paddleformers.transformers import ShieldGemma2Processor
    processor = ShieldGemma2Processor.from_pretrained(str(Path(checkpoint).resolve()))
    module = importlib.import_module(type(processor).__module__)
    require(Path(module.__file__).resolve().is_relative_to(Path(source).resolve()), "processor source differs from locked source")
    results = []
    for split in ("train", "eval"):
        entry = data_report["splits"][split]["entries"][0]
        with Image.open(PROJECT / entry["image"]) as image:
            encoded = processor(images=[image.convert("RGB")], policies=[entry["policy"]], return_tensors="np",
                                images_kwargs={"do_pan_and_scan": False},
                                text_kwargs={"padding": "max_length", "padding_side": "left", "max_length": 448, "truncation": False})
        arrays = {name: np.asarray(encoded[name], dtype=np.float32 if name == "pixel_values" else np.int64)
                  for name in INPUT_NAMES if name != "position_ids"}
        arrays["position_ids"] = np.maximum(np.cumsum(arrays["attention_mask"], axis=-1) - 1, 0).astype(np.int64)
        comparisons = {}
        for name in INPUT_NAMES:
            first, second = arrays[name], loaded[split][0][name]
            same = first.shape == second.shape and first.dtype == second.dtype and np.array_equal(first, second)
            require(same, f"independent clean processor differs for {split}/{name}")
            comparisons[name] = {"equal": True, "shape": list(first.shape), "dtype": str(first.dtype), "max_abs_difference": 0.0}
        results.append({"split": split, "batch": entry["batch"], "inputs": comparisons,
                        "labels": "original NPZ/provenance checked separately; processor does not invent safety labels"})
    return {"passed": True, "independent_processor_instances": 1, "rebuilt_train_batches": 1, "rebuilt_eval_batches": 1,
            "all_five_inputs_exact": True, "processor_source": reference(module.__file__), "comparisons": results}


def cpu_model_reference(source, checkpoint, batch):
    paddle = clean_imports(source)
    from paddle.base import core
    from paddleformers.transformers import ShieldGemma2ForImageClassification
    core.set_cublas_switch(False)
    core.set_cudnn_switch(False)
    started = time.monotonic()
    model = ShieldGemma2ForImageClassification.from_pretrained(str(Path(checkpoint).resolve()), dtype="float32")
    model.eval()
    require(model.get_output_embeddings().weight is model.get_input_embeddings().weight, "original classifier head is not tied")
    require({str(parameter.dtype) for parameter in model.parameters()} == {"paddle.float32"}, "model parameters are not FP32")
    sys.path.insert(0, str(MODEL))
    from classification_graph import ShieldGemma2InferGraph
    graph = ShieldGemma2InferGraph(model).eval()
    tensors = {name: paddle.to_tensor(batch[name]) for name in INPUT_NAMES}
    head_shapes = []
    def record_head(layer, arguments, output):
        head_shapes.append(list(output.shape))
    handle = model.lm_head.register_forward_post_hook(record_head)
    with paddle.no_grad():
        native = model(**tensors, use_cache=False, return_dict=True, logits_to_keep=0)
    handle.remove()
    with paddle.no_grad():
        logits, probabilities = graph(*(tensors[name] for name in INPUT_NAMES))
    require(head_shapes == [[1, 448, 262208]], "native reference did not compute full-vocabulary logits")
    differences = {}
    for name, first, second, tolerance in (("logits", native.logits, logits, 1e-3),
                                           ("probabilities", native.probabilities, probabilities, 1e-4)):
        left, right = first.numpy(), second.numpy()
        require(np.isfinite(left).all() and np.isfinite(right).all(), "nonfinite native/optimized reference")
        maximum = float(np.max(np.abs(left - right)))
        require(maximum <= tolerance and np.array_equal(left.argmax(-1), right.argmax(-1)), "native full-vocab and two-column graph differ")
        differences[name] = {"max_abs_difference": maximum, "atol": tolerance, "rtol": 0,
                             "native": left.tolist(), "optimized": right.tolist()}
    return {"passed": True, "device": "cpu", "dtype": "float32", "amp": False, "tf32": False,
            "batch": "train:0", "native_lm_head_shapes": head_shapes, "head_shared": True,
            "classification_graph": reference(MODEL / "classification_graph.py"),
            "comparisons": differences, "elapsed_s": time.monotonic() - started,
            "scope": "one real image-policy pair structural output equivalence; no safety accuracy gate"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--source-lock", type=Path, default=SOURCE_LOCK)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--model-reference", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output must be fresh")
    require(args.output.resolve().is_relative_to(PROJECT), "output must stay inside CINN")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU asset validation requires CUDA_VISIBLE_DEVICES=''")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "8"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for name in ("FLAGS_use_cinn", "FLAGS_prim_all", "FLAGS_prim_enable_dynamic",
                 "FLAGS_enable_cublas_tensor_op_math", "FLAGS_enable_cudnn_tensor_op_math"):
        os.environ[name] = "false"
    started = time.monotonic()
    snapshot = source_snapshot(args.source, args.source_lock)
    print("ASSETS source_lock_verified " + snapshot["source_commit"], flush=True)
    checkpoint = audit_checkpoint(args.checkpoint)
    print("ASSETS ten_checkpoint_hashes_and_67_BF16_tensors_verified", flush=True)
    data, loaded = audit_data(args.checkpoint, args.data, args.images)
    print("ASSETS eight_disjoint_images_and_fixed_batches_verified", flush=True)
    processor = rebuild_processor(args.source, args.checkpoint, data, loaded)
    print("ASSETS clean_train_eval_processor_all_five_inputs_exact", flush=True)
    model = cpu_model_reference(args.source, args.checkpoint, loaded["train"][0]) if args.model_reference else {
        "executed": False, "reason": "optional CPU full-vocabulary reference not requested; no numerical claim from historical source"}
    require(source_snapshot(args.source, args.source_lock) == snapshot, "source changed during asset verification")
    result = {"schema": "shieldgemma2_v2_assets_v1", "passed": True, "path_base": "CINN_project_root",
              "source": snapshot, "source_snapshot": snapshot, "checkpoint": checkpoint, "data": data, "processor": processor,
              "model_reference": model, "assets_script": reference(__file__), "elapsed_s": time.monotonic() - started,
              "device": "cpu", "gpu_executed": False, "threads": 8, "cuda_visible_devices": "",
              "scope": "real text2/vision2 structural consistency; existing AI-reviewed labels preserved, not gold safety supervision",
              "safety_accuracy_gate": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print("ASSETS_VERIFIED " + json.dumps(reference(args.output)), flush=True)


if __name__ == "__main__":
    main()
