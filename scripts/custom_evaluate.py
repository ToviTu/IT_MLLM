from torch.utils.data import Dataset
import json
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from scripts.download_dataset import DATASET_DIR
from scripts.download_dataset import DATASET_URL
from open_flamingo.eval.eval_datasets import VQADataset
from open_flamingo.eval.vqa_metric import (
    compute_vqa_accuracy,
    postprocess_vqa_generation,
)
from open_flamingo.eval.eval_model import BaseEvalModel
from open_flamingo.eval.models.open_flamingo import EvalModel
import scripts.utils as utils


def evaluate_vqa(
    eval_model: BaseEvalModel,
    seed: int = 42,
    min_generation_length: int = 0,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = 0.0,
    num_shots: int = 8,
    dataset_name: str = "vqav2",
    query_set_size=2048,
    num_samples=1000,
    batch_size=8,
):
    train_image_dir_path = DATASET_DIR + "train2014"
    train_questions_json_path = (
        DATASET_DIR + "v2_OpenEnded_mscoco_train2014_questions.json"
    )
    train_annotations_json_path = DATASET_DIR + "v2_mscoco_train2014_annotations.json"
    test_image_dir_path = DATASET_DIR + "val2014"
    test_questions_json_path = (
        DATASET_DIR + "v2_OpenEnded_mscoco_val2014_questions.json"
    )
    test_annotations_json_path = DATASET_DIR + "v2_mscoco_val2014_annotations.json"

    train_dataset = VQADataset(
        image_dir_path=train_image_dir_path,
        question_path=train_questions_json_path,
        annotations_path=train_annotations_json_path,
        is_train=True,
        dataset_name=dataset_name,
    )

    test_dataset = VQADataset(
        image_dir_path=test_image_dir_path,
        question_path=test_questions_json_path,
        annotations_path=test_annotations_json_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    effective_num_shots = utils.compute_effective_num_shots(num_shots, "OpenFlamingo")

    np.random.seed(seed)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=utils.custom_collate_fn
    )

    query_set = utils.get_query_set(train_dataset, query_set_size)

    counter = 0
    utils.random_seed(seed)
    predictions = []
    for batch in tqdm(
        test_dataloader,
        desc=f"Running inference {dataset_name}",
    ):
        if counter >= num_samples:
            break
        counter += 1

        batch_demo_samples = utils.sample_batch_demos_from_query_set(
            query_set, effective_num_shots, len(batch["image"])
        )

        batch_images, batch_text = [], []
        for i in range(len(batch["image"])):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            batch_images.append(context_images + [batch["image"][i]])

            context_text = "".join(
                [
                    eval_model.get_vqa_prompt(
                        question=x["question"], answer=x["answers"][0]
                    )
                    + "\n"
                    for x in batch_demo_samples[i]
                ]
            )

            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(
                context_text + eval_model.get_vqa_prompt(question=batch["question"][i])
            )

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            min_generation_length=min_generation_length,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        process_function = postprocess_vqa_generation

        new_predictions = map(process_function, outputs)

        for new_prediction, sample_id in zip(new_predictions, batch["question_id"]):
            predictions.append({"answer": new_prediction, "question_id": sample_id})

    with open(f"{dataset_name}results.json", "w") as f:
        f.write(json.dumps(predictions, indent=4))

    acc = -1
    if test_annotations_json_path is not None:
        acc = compute_vqa_accuracy(
            f"{dataset_name}results.json",
            test_questions_json_path,
            test_annotations_json_path,
        )
        # delete the temporary file
        os.remove(f"{dataset_name}results.json")

    return acc
