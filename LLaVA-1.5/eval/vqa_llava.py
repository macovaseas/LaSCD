import argparse
import torch
import os
import json
from tqdm import tqdm
import requests
from io import BytesIO
from typing import Dict, List, Optional, Tuple
import shortuuid
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images

from PIL import Image
import math
import re
# import kornia
from transformers import set_seed


DATASET_DEFAULTS = {
    "mmvet": {"conv_mode": "llava_v1", "one_word_suffix": True},
    "llavabench": {"conv_mode": "llava_v1", "one_word_suffix": True},
    "vizwiz": {"conv_mode": "vicuna_v1", "one_word_suffix": False},
    "hallusionbench": {"conv_mode": "llava_v1", "one_word_suffix": False},
}

TEXT_ONLY_PLACEHOLDER_SIZE = (224, 224)
TEXT_ONLY_PLACEHOLDER_COLOR = (128, 128, 128)


def _split_hallusion_rel(filename: str) -> Tuple[str, str]:
    rel = str(filename).replace("\\", "/").strip()
    if rel.startswith("./"):
        rel = rel[2:]
    under = rel
    prefix = "hallusion_bench/"
    if under.lower().startswith(prefix):
        under = under[len(prefix) :]
    return rel, under


def _resolve_image_path(
    image_root: str,
    dataset_root: str,
    sample: Dict,
) -> Optional[str]:
    filename = sample.get("filename")

    candidates: List[str] = []
    image_root = os.path.abspath(os.path.expanduser(image_root)) if image_root else ""
    dataset_root = os.path.abspath(os.path.expanduser(dataset_root)) if dataset_root else ""

    if filename:
        rel, under = _split_hallusion_rel(filename)
        if image_root:
            candidates.append(os.path.join(image_root, under))
            if under != rel:
                candidates.append(os.path.join(image_root, rel))
        if dataset_root:
            candidates.append(os.path.join(dataset_root, rel))
            candidates.append(os.path.join(dataset_root, "hallusion_bench", under))
    elif str(sample.get("visual_input", "")) in ("1", "2"):
        cat = str(sample.get("category", "")).strip()
        sub = str(sample.get("subcategory", "")).strip()
        sid = str(sample.get("set_id", "")).strip()
        fid = str(sample.get("figure_id", "")).strip()
        if cat and sub and sid and fid:
            under = os.path.join(cat, sub, f"{sid}_{fid}.png")
            if image_root:
                candidates.append(os.path.join(image_root, under))
            if dataset_root:
                candidates.append(os.path.join(dataset_root, "hallusion_bench", under))

    seen = set()
    ordered: List[str] = []
    for p in candidates:
        p = os.path.normpath(p)
        if p not in seen:
            seen.add(p)
            ordered.append(p)

    for p in ordered:
        if os.path.isfile(p):
            return p
    return ordered[0] if ordered else None


class SafeKeywordsStoppingCriteria(KeywordsStoppingCriteria):
    """Avoid matching stop keywords against pure prompt (no newly generated token)."""

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if output_ids.shape[1] <= self.start_len:
            return False
        return super().__call__(output_ids, scores, **kwargs)


def get_chunk(lst, n, k):
    chunk = math.ceil(len(lst) / n)
    return lst[k * chunk : (k + 1) * chunk]


def build_user_message(qs_base: str, one_word_suffix: bool) -> str:
    if one_word_suffix:
        return qs_base + " Please answer this question with one word."
    return qs_base


def _load_image_for_hallusion(sample: Dict, image_folder: str, dataset_root: str) -> Image.Image:
    vi = str(sample.get("visual_input", "1"))
    if vi == "0":
        return Image.new("RGB", TEXT_ONLY_PLACEHOLDER_SIZE, TEXT_ONLY_PLACEHOLDER_COLOR)
    image_path = _resolve_image_path(image_folder, dataset_root, sample)
    if image_path is None or not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found for question_id={sample.get('question_id')!r}: {image_path}"
        )
    return Image.open(image_path).convert("RGB")


def eval_model(args):
    device = args.cuda_device
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, _ctx = load_pretrained_model(model_path, args.model_base, model_name,device="cuda")
    print(type(model))

    ds = DATASET_DEFAULTS[args.dataset]
    conv_mode = args.conv_mode or ds["conv_mode"]
    one_word = ds["one_word_suffix"] if not args.ignore_dataset_prompt else False

    if args.dataset == "hallusionbench":
        dataset_file = os.path.expanduser(args.dataset_file)
        with open(dataset_file, "r", encoding="utf-8") as f:
            questions = json.load(f)
        n_full = len(questions)
        if args.only_visual_input_1:
            questions = [x for x in questions if str(x.get("visual_input", "1")) == "1"]
            print(
                f"警告: --only-visual-input-1 已启用，仅 {len(questions)}/{n_full} 条；"
                "与 evaluation.py 全量评测不兼容。",
                flush=True,
            )
        if args.limit > 0:
            questions = questions[: args.limit]
        dataset_root = args.dataset_root or os.path.dirname(os.path.abspath(dataset_file))
        image_folder = os.path.expanduser(args.image_folder)
        answers_file = os.path.expanduser(args.answers_file)
        os.makedirs(os.path.dirname(answers_file) or ".", exist_ok=True)
        results: List[Dict] = []
    else:
        questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
        dataset_root = ""
        image_folder = os.path.expanduser(args.image_folder)
        answers_file = os.path.expanduser(args.answers_file)
        os.makedirs(os.path.dirname(answers_file) or ".", exist_ok=True)

    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in conv_mode:
        conv_mode = conv_mode + "_mmtag"
        print(f"Auto switch conv-mode -> {conv_mode}")

    hallusion_results_mode = args.dataset == "hallusionbench"
    ans_file = None if hallusion_results_mode else open(answers_file, "w")

    try:
        iterator = tqdm(questions, desc=f"vqa_llava/{args.dataset}")
        for line in iterator:
            if hallusion_results_mode:
                qs_text = line["question"]
                if args.constrain_yn:
                    qs_text = qs_text.rstrip() + "\nAnswer with exactly one word: Yes, No, or Uncertain."
                try:
                    image = _load_image_for_hallusion(line, image_folder, dataset_root)
                except Exception as exc:
                    outputs = f"[ERROR] {type(exc).__name__}: {exc}"
                    out_row = dict(line)
                    out_row["model_prediction"] = outputs
                    results.append(out_row)
                    if len(results) % args.save_every == 0:
                        with open(answers_file, "w", encoding="utf-8") as f:
                            json.dump(results, f, ensure_ascii=False)
                    continue
            else:
                idx = line["question_id"]
                image_file = line["image"]
                qs_text = line["text"]
                cur_prompt = qs_text
                image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")

            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs_text
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs_text

            user_turn = build_user_message(qs, one_word)
            conv = conv_templates[conv_mode].copy()
            conv.append_message(conv.roles[0], user_turn)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0)
            input_ids = input_ids.to(device)
            image_tensor = process_images([image], image_processor, model.config)[0]
            img_batch = image_tensor.unsqueeze(0).half().to(device)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            gen_kw = dict(
                images=img_batch,
                # image_sizes=[image.size],
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
            if args.use_lascd:
                gen_kw["stopping_criteria"] = [
                    SafeKeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
                ]
            if args.top_p is not None:
                gen_kw["top_p"] = args.top_p
            if args.top_k is not None:
                gen_kw["top_k"] = args.top_k

            try:
                with torch.inference_mode():
                    if args.use_lascd:
                        out = model.generate(
                            input_ids,
                            use_lascd=True,
                            alpha=args.alpha,
                            threshold_top_p=args.threshold_top_p,
                            threshold_top_k=args.threshold_top_k,
                            early_exit_layers=list(range(args.start_layer, args.end_layer)),
                            laplacian_mode=args.laplacian_mode,
                            beta=args.beta,
                            energy_mode=args.energy_mode,
                            force_1d=args.force_1d,
                            output_hidden_states=True,
                            return_dict_in_generate=True,
                            return_dict=True,
                            **gen_kw,
                        )
                        output_ids = out.sequences
                    else:
                        output_ids = model.generate(
                            input_ids,
                            use_cache=True,
                            **gen_kw,
                        )

                in_len = input_ids.shape[1]
                if (
                    output_ids.shape[1] > in_len
                    and torch.equal(output_ids[:, :in_len], input_ids)
                ):
                    decode_ids = output_ids[:, in_len:]
                else:
                    decode_ids = output_ids

                outputs = tokenizer.batch_decode(decode_ids, skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[: -len(stop_str)].strip()
            except Exception as exc:
                if hallusion_results_mode:
                    outputs = f"[ERROR] {type(exc).__name__}: {exc}"
                else:
                    raise

            if hallusion_results_mode:
                out_row = dict(line)
                out_row["model_prediction"] = outputs
                results.append(out_row)
                if len(results) % args.save_every == 0:
                    with open(answers_file, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False)
            else:
                meta = {"dataset": args.dataset, "use_lascd": args.use_lascd}
                if "category" in line:
                    meta["category"] = line["category"]

                ans_file.write(
                    json.dumps(
                        {
                            "question_id": idx,
                            "prompt": cur_prompt,
                            "text": outputs,
                            "answer_id": shortuuid.uuid(),
                            "model_id": model_name,
                            "metadata": meta,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                ans_file.flush()
    finally:
        if ans_file is not None:
            ans_file.close()

    if hallusion_results_mode:
        with open(answers_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

    print(f"Done -> {answers_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LLaVA 通用 VQA（mmvet / vizwiz / llavabench / hallusionbench）")
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_DEFAULTS.keys()),
        help="hallusionbench：HallusionBench.json + 与 evaluation.py 一致的 JSON 输出",
    )
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--model-base", type=str, default=None)
    p.add_argument(
        "--image-folder",
        type=str,
        default=None,
        help="图片目录：jsonl 时为每条的 image 相对路径前缀；hallusionbench 时为 hallusion_bench 根目录（与 qwen3.py --image-root 一致）",
    )
    p.add_argument(
        "--question-file",
        type=str,
        default=None,
        help="jsonl 问题文件（非 hallusionbench 时必填）",
    )
    p.add_argument(
        "--answers-file",
        type=str,
        required=True,
        help="输出路径：jsonl（通用 VQA）或 JSON 数组（hallusionbench，供 evaluation.py 使用）",
    )
    p.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="HallusionBench.json（仅 dataset=hallusionbench 时必填）",
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="HallusionBench 仓库根目录（解析 ./hallusion_bench/...）；默认取 dataset-file 所在目录",
    )
    p.add_argument("--conv-mode", type=str, default=None, help="覆盖数据集默认 conv 模板")
    p.add_argument(
        "--ignore-dataset-prompt",
        action="store_true",
        help="不追加 one-word 后缀（即使数据集默认为 mmvet/llavabench）",
    )
    p.add_argument(
        "--constrain-yn",
        action="store_true",
        help="HallusionBench：在问题后追加 Yes/No/Uncertain 单词约束",
    )
    p.add_argument("--limit", type=int, default=0, help="HallusionBench：仅跑前 N 条；0 表示全部")
    p.add_argument("--only-visual-input-1", action="store_true", help="HallusionBench：仅 visual_input==1")
    p.add_argument("--save-every", type=int, default=20, help="HallusionBench：每 N 条落盘一次（断点近似恢复）")
    p.add_argument("--num-chunks", type=int, default=1)
    p.add_argument("--chunk-idx", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--top_k", type=int, default=None)
    p.add_argument("--num_beams", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--cuda-device", type=str, default="cuda:0")
    p.add_argument("--use-lascd", action="store_true")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--threshold_top_p", type=float, default=0.9)
    p.add_argument("--threshold_top_k", type=int, default=20)
    p.add_argument("--start_layer", type=int, default=15)
    p.add_argument("--end_layer", type=int, default=20)
    p.add_argument("--laplacian_mode", type=str, default="standard", choices=["standard", "sobel", "log"])
    p.add_argument("--energy_mode", type=str, default="vision", choices=["vision", "text"])
    p.add_argument("--force_1d", action="store_true")
    p.add_argument("--apply-memvr", type=str, default="default")
    p.add_argument("--retracing-ratio", type=float, default=0.0)
    p.add_argument("--entropy-threshold", type=float, default=0.75)
    p.add_argument("--starting-layer", type=int, default=5)
    p.add_argument("--ending-layer", type=int, default=16)
    args = p.parse_args()

    if args.dataset == "hallusionbench":
        if not args.dataset_file:
            p.error("hallusionbench 需要 --dataset-file 指向 HallusionBench.json")
        if not args.image_folder:
            p.error("hallusionbench 需要 --image-folder 指向 hallusion_bench 图片根目录")
    else:
        if not args.question_file:
            p.error(f"{args.dataset} 需要 --question-file")
        if not args.image_folder:
            p.error(f"{args.dataset} 需要 --image-folder")

    eval_model(args)
