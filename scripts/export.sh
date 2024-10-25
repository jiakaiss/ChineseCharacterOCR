export CUDA_VISIBLE_DEVICES=0

python export/export.py \
        --model_path "output/evaluate15-2/model.pth" \
        --out_path export/ocr \
        --input_shape 1 3 64 64 \
        --use_evaluate