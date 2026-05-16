import argparse
import torch
import os
import json
import random
from tqdm import tqdm
import shortuuid
import os
import nltk
# nltk.download('punkt_tab')
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import (
    IMAGE_TOKEN_INDEX, # IMAGE_TOKEN_INDEX = -200
    DEFAULT_IMAGE_TOKEN, # DEFAULT_IMAGE_TOKEN = "<image>"
    DEFAULT_IM_START_TOKEN, # DEFAULT_IM_START_TOKEN = "<im_start>"
    DEFAULT_IM_END_TOKEN, # DEFAULT_IM_END_TOKEN = "<im_end>"
    IMAGE_PLACEHOLDER, # IMAGE_PLACEHOLDER = "<image-placeholder>"
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import base64
import requests

from io import BytesIO
import re
import math
import torch.distributed as dist
from utils import dist_util
from utils.logger import create_logger
from glob import glob
from transformers import set_seed


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def eval_model(args):

    # set up gpu and logging
    dist_util.setup_dist(args)
    device = dist_util.device()

    # Setup an experiment folder:
    if dist.get_rank() == 0:
        os.makedirs(
            args.log_path, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        model_string_name = args.model_path.split("/")[-1]
        experiment_index = len(glob(f"{args.log_path}/{model_string_name}/*"))
        experiment_dir = f"{args.log_path}"  # Create an experiment folder
        os.makedirs(experiment_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device="cuda:0"
    )
    

    img_files = os.listdir(args.image_folder)
    random.shuffle(img_files)

    with open(args.question_file, "r") as f:
        lines = f.readlines()
    coco_anns = json.loads(lines[0])

    img_dict = {}

    categories = coco_anns["categories"]
    category_names = [c["name"] for c in categories]
    category_dict = {int(c["id"]): c["name"] for c in categories}

    for img_info in coco_anns["images"]:
        img_dict[img_info["id"]] = {"name": img_info["file_name"], "anns": []}

    for ann_info in coco_anns["annotations"]:
        img_dict[ann_info["image_id"]]["anns"].append(
            category_dict[ann_info["category_id"]]
        )

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    with open(answers_file, "w", encoding="utf-8") as _f:
        pass

    # =========================================================
    # iterate over shuffled image folder, cap at 500
    # =========================================================
    total = min(500, len(img_files))
    for i in tqdm(range(len(img_files)), total=total):
        if i == 500:
            break
        img_file = img_files[i]
        idx = int(img_file.split(".jpg")[0][-6:])
        img_info = img_dict[idx]
        assert img_info["name"] == img_file
        image_file = os.path.join(args.image_folder, img_file)

        qs = "Please describe this image in detail."  # line["query"]
        cur_prompt = qs

        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        image = Image.open(image_file)
        image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        with torch.inference_mode():
            with torch.no_grad():
                output_dict = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    stopping_criteria=[stopping_criteria],
                    use_lascd=True,
                    alpha=args.alpha,
                    threshold_top_p=args.threshold_top_p,
                    threshold_top_k=args.threshold_top_k,
                    early_exit_layers=[i for i in range(args.start_layer, args.end_layer)],
                    laplacian_mode=args.laplacian_mode,
                    beta=args.beta,
                    energy_mode=args.energy_mode,
                    force_1d=args.force_1d,
                    layer_select_metric="energy",
                    return_dict=True,
                    debug=False,
                )

        output_ids = output_dict.sequences
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()

        logger.info(f"[{image_file}]")
        logger.info(f"prompt: {cur_prompt}")
        logger.info(f"text: {outputs}")

        res_dict = {"image_id": idx, "caption": outputs}
        with open(answers_file, "a", encoding="utf-8") as ans_file:
            ans_file.write(json.dumps(res_dict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/hpc2hdd/home/fcao628/models/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str,
                        default="/hpc2hdd/home/fcao628/dataset/val2014",
                        help="COCO val2014 image folder")
    parser.add_argument("--question-file", type=str,
                        default="/hpc2hdd/home/fcao628/LLaVA/playground/data/eval/chair/annotations/instances_val2014.json",
                        help="COCO annotations json (categories + images + annotations)")
    parser.add_argument("--answers-file", type=str,
                        default="/hpc2hdd/home/fcao628/LaSCD/LLaVA-1.5/output/chair/chair_result.jsonl")
    # ---
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--log_path", type=str, default="/hpc2hdd/home/fcao628/LaSCD/LLaVA-1.5/log")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--threshold_top_p", type=float, default=0.9)
    parser.add_argument("--threshold_top_k", type=int, default=20)
    parser.add_argument("--start_layer", type=int, default=11)
    parser.add_argument("--end_layer", type=int, default=29)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--laplacian_mode", type=str, default="standard", choices=["standard", "sobel", "log"])
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--energy_mode", type=str, default="vision", choices=["vision", "text"])
    parser.add_argument("--force_1d", action="store_true",
                        help="Force 1-D Laplacian in standard mode even for square grids (ablation)")

    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
