IMAGE_NAME=tovitu/lang-modeling

build:
	docker build . -t ${IMAGE_NAME} 
push:
	docker push ${IMAGE_NAME}
clean: 
	docker system prune -af
eval_vicuna:
	python -m llava.eval.model_vqa_loader \
    --model-path /scratch/vision_share/models/llava-llama2-7b-pretrain \
    --model-base meta-llama/Llama-2-7b-hf \
    --question-file ${EVAL_DIR}/vizwiz/llava_test.jsonl \
    --image-folder ${EVAL_DIR}/vizwiz/test \
    --answers-file ${EVAL_DIR}/vizwiz/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode v1

