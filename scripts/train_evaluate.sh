export CUDA_VISIBLE_DEVICES=0

python main_evaluate.py \
        --dataset-path "C:\MyDataset\handwritten_check" \
        --background-path "C:\MyAiProject\ChineseCharacterOCR\dataset\copybook_background" \
        --standard-path "C:\MyDataset\standard_images" \
        --good-path "C:\MyDataset\handwritten_judge\handwritten_good" \
        --medium-path "C:\MyDataset\handwritten_judge\handwritten_medium" \
        --bad-path "C:\MyDataset\handwritten_judge\handwritten_bad" \
        --output-path output/evaluate15-2 \
        --pretrained-path "output/classify-9663/model.pth" \
        --margin 15