python -m eval.pope_llava \
  --image-folder /hpc2hdd/home/fcao628/dataset/POPEv2 \
  --question-file /hpc2hdd/home/fcao628/VCD/experiments/data/POPE/v2/POPEv2.json \
  --answers-file output/popev2.json \
  --conv-mode llava_v1 \
  --temperature 0 \
  --max_new_tokens 1 \
  --top_p 0.9 \
  --alpha 0.1 \
  --start_layer 2 \
  --end_layer 15 \
  --threshold_top_k 10 \
  --laplacian_mode standard \
  --beta 0.0 \
  --energy_mode vision
  # --force_1d  # uncomment to ablate 1-D vs 2-D Laplacian in standard mode



python /hpc2hdd/home/fcao628/LaSCD/eval_tool/eval_pope.py \
  --gt_files /hpc2hdd/home/fcao628/VCD/experiments/data/POPE/v2/POPEv2.json \
  --gen_files output/popev2.json