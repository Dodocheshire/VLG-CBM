import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

import data.utils as utils
import GroundingDINO.groundingdino.datasets.transforms as T
from data.utils import get_data
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Resize(object):
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize((size, size))

    def __call__(self, img, target):
        return self.resize(img), target


def load_annotation_model(model_config_path, model_checkpoint_path, device="cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    local_text_encoder = Path(__file__).resolve().parents[1] / "bert-base-uncased"
    args.text_encoder_type = os.environ.get(
        "GROUNDING_DINO_TEXT_ENCODER",
        str(local_text_encoder) if local_text_encoder.is_dir() else args.text_encoder_type,
    )
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    logger.info(f"Loaded Grounding DINO on {device}: {load_res}")
    model.eval()
    model.to(device)
    tokenlizer = model.tokenizer
    return model, tokenlizer


def get_predictions(model: Any, images_tensor: torch.Tensor, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(images_tensor, captions=prompts)
    logits = outputs["pred_logits"].sigmoid()
    boxes = outputs["pred_boxes"]
    return logits, boxes


def process_annotations_for_bbox(
    prompt: str,
    bbox: np.array,
    prompt_logits: np.array,
    tokenlizer: Any,
    text_threshold: float,
) -> List[Dict]:
    assert len(prompt_logits.shape) == 1
    prompt_logits = prompt_logits[1:-1]
    prompt_tokenized = tokenlizer(prompt)
    prompt_tokenized = prompt_tokenized["input_ids"][1:-1]

    split_token_idxs = [i for i, x in enumerate(prompt_tokenized) if x == 1012]
    split_token_idxs = [-1] + split_token_idxs

    phrases = []
    for i in range(len(split_token_idxs) - 1):
        concept_token_idxs = prompt_tokenized[split_token_idxs[i] + 1 : split_token_idxs[i + 1]]
        concept_logits = prompt_logits[split_token_idxs[i] + 1 : split_token_idxs[i + 1]]
        if len(concept_logits) == 0:
            continue
        concept = tokenlizer.decode(concept_token_idxs).strip()
        score = np.max(concept_logits)
        if score > text_threshold:
            phrases.append({
                "concept": concept,
                "score": float(score),
                "bbox": [float(x) for x in bbox],
            })
    return phrases


def process_annotations(
    image_pil: Image.Image,
    prompt: str,
    logits_image: np.array,
    bboxes_image: np.array,
    tokenlizer: Any,
    text_threshold: float,
) -> List[Dict]:
    w, h = image_pil.size
    annotations = []
    for logit, bbox in zip(logits_image, bboxes_image):
        scaled_bbox = bbox * np.array([w, h, w, h])
        scaled_bbox = scaled_bbox.tolist()
        scaled_bbox = [scaled_bbox[0] - scaled_bbox[2] / 2, scaled_bbox[1] - scaled_bbox[3] / 2, scaled_bbox[0] + scaled_bbox[2] / 2, scaled_bbox[1] + scaled_bbox[3] / 2]
        phrases = process_annotations_for_bbox(prompt, scaled_bbox, logit, tokenlizer, text_threshold)
        for phrase in phrases:
            annotations.append({
                "label": phrase["concept"],
                "box": phrase["bbox"],
                "logit": phrase["score"],
            })
    return annotations


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Grounding DINO Annotation Generator")
    parser.add_argument("--config_file", type=str, default="GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py")
    parser.add_argument("--grounded_checkpoint", type=str, default="GroundingDINO/groundingdino_swinb_cogcoor.pth")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--text_threshold", type=float, default=0.15)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--start_class_idx", type=int, default=None)
    parser.add_argument("--end_class_idx", type=int, default=None)
    parser.add_argument(
        "--max_images_per_class",
        type=int,
        default=None,
        help="Optional cap used for smoke tests",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_prefix = args.dataset_name.split("_")[0]
    classes = utils.get_classes(dataset_prefix)
    per_class_concepts_file = f"concept_files/{dataset_prefix}_per_class.json"
    with open(per_class_concepts_file, "r") as f:
        per_class_concepts = json.load(f)

    logger.info(f"Loading dataset {args.dataset_name} on {args.device}...")
    pil_data = utils.get_data(args.dataset_name, preprocess=None)
    transform = T.Compose([
        Resize(800),
        T.RandomResize([800]),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = get_data(args.dataset_name, preprocess=lambda x: transform(x, None)[0])

    model, tokenlizer = load_annotation_model(args.config_file, args.grounded_checkpoint, device=args.device)

    start_cls = args.start_class_idx if args.start_class_idx is not None else 0
    end_cls = args.end_class_idx if args.end_class_idx is not None else len(classes)
    logger.info(f"[{args.device}] Running on classes {start_cls} to {end_cls} (Total: {end_cls - start_cls} classes)")

    total_annotated = 0
    start_time = time.time()

    for class_idx in range(start_cls, end_cls):
        class_name = classes[class_idx]
        per_class_concept = per_class_concepts[class_name]

        prompt = utils.format_concept(class_name) + " . "
        for concept in per_class_concept:
            prompt = prompt + f"{utils.format_concept(concept)} . "
        prompt = prompt.strip()

        class_indices = np.where(np.array(dataset.targets) == class_idx)[0]
        if args.max_images_per_class is not None:
            class_indices = class_indices[: args.max_images_per_class]
        if len(class_indices) == 0:
            continue

        # Check if all indices in this class already exist
        missing_indices = [idx for idx in class_indices if not os.path.exists(os.path.join(args.output_dir, f"{idx}.json"))]
        if len(missing_indices) == 0:
            continue

        subset = torch.utils.data.Subset(dataset, missing_indices)
        dataloader = torch.utils.data.DataLoader(
            subset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True,
        )

        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(args.device)
            logits, boxes = get_predictions(model, images, [prompt] * images.shape[0])

            for image_idx in range(logits.shape[0]):
                global_idx = subset.indices[batch_idx * args.batch_size + image_idx]
                out_path = os.path.join(args.output_dir, f"{global_idx}.json")
                if os.path.exists(out_path):
                    continue

                image_pil = pil_data[global_idx][0]
                logits_img = logits[image_idx].clone().cpu().numpy()
                boxes_img = boxes[image_idx].clone().cpu().numpy()
                annotations = process_annotations(image_pil, prompt, logits_img, boxes_img, tokenlizer, args.text_threshold)

                # Save JSON
                img_path = getattr(dataset, "imgs", None)
                if img_path is not None and global_idx < len(img_path):
                    actual_path = img_path[global_idx]
                else:
                    actual_path = None

                data_to_save = [{"path": actual_path}] + annotations
                with open(out_path, "w") as f:
                    json.dump(data_to_save, f, indent=2)

                total_annotated += 1

        elapsed = time.time() - start_time
        speed = total_annotated / max(1, elapsed)
        logger.info(f"[{args.device}] Completed class {class_idx}/{end_cls} ({class_name}): total {total_annotated} imgs annotated ({speed:.1f} img/s)")

    peak_gib = torch.cuda.max_memory_allocated(args.device) / (1024 ** 3)
    logger.info(
        f"[{args.device}] Done! Total annotated: {total_annotated} images. "
        f"Peak allocated VRAM: {peak_gib:.2f} GiB"
    )

if __name__ == "__main__":
    main()
