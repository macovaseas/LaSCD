python eval/chair_llava.py \
    --temperature 0 \
    --max_new_tokens 1024 \
    --top_p 0.9 \
    --alpha 0.1 \
    --start_layer 11 \
    --end_layer 29 \
    --threshold_top_k 10 \
    --laplacian_mode standard \
    --energy_mode vision \
    --beta 0.6 \
    --seed 2026
    # --force_1d  # uncomment to ablate 1-D vs 2-D Laplacian in standard mode

python eval/chair.py \
    --cap_file /hpc2hdd/home/fcao628/LaSCD/LLaVA-1.5/output/chair/chair_result.jsonl \
    --image_id_key image_id \
    --caption_key caption \
    --coco_path /hpc2hdd/home/fcao628/LLaVA/playground/data/eval/chair/annotations \
    --save_path output/chair/chair.jsonl