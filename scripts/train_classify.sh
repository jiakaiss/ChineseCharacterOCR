export CUDA_VISIBLE_DEVICES=0

python main.py \
        --dataset-path C:\\MyAiProject\\ChineseCharacterOCR\\dataset \
        --background-path C:\\MyAiProject\\ChineseCharacterOCR\\dataset\\copybook_background \
        --output-path output/classify3-nocl \
        --pretrained-path "output/pretrain_512dim_model.pth" \
        --lr-model 0.0005 \
        # --use-centerloss