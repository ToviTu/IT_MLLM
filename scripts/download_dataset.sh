#! /bin/bash

OUTPUT_DIR="${STORAGE_DIR}dataset/"
EVAL_DIR="${STORAGE_DIR}dataset/eval"

# Download VQAv2 Train Split
# wget -P $OUTPUT_DIR https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
# wget -P $OUTPUT_DIR https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip

# unzip -d ${OUTPUT_DIR} ${OUTPUT_DIR}v2_Questions_Train_mscoco.zip
# unzip -d ${OUTPUT_DIR} ${OUTPUT_DIR}v2_Annotations_Train_mscoco.zip

# Download StrategyQA
# wget -P $OUTPUT_DIR https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip
# unzip -d $OUTPUT_DIR ${OUTPUT_DIR}strategyqa_dataset.zip

# Download ARC
# wget -P $OUTPUT_DIR https://ai2-public-datasets.s3.amazeonaws.com/arc/ARC-V1-Feb2018.zip
# unzip -d $OUTPUT_DIR ${OUTPUT_DIR}ARC-V1-Feb20

# Download CommonsenseQA
# wget -P $OUTPUT_DIR https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl

# # Download LLaVA pretraining dataset
# wget -P $OUTPUT_DIR https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/blip_laion_cc_sbu_558k.json
# #wget -P $OUTPUT_DIR https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/resolve/main/images.zip
# wget -P $OUTPUT_DIR https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/resolve/main/images.zip
# unzip -d ${OUTPUT_DIR}images/ ${OUTPUT_DIR}images.zip

# Updated on Jun.18th
# Download Evaluation Benchmarks for LLaVA v1.5

# Download Scripts (containing custom annotations, scripts, and the prediction files with LLaVA v1.5)
wget -P $EVAL_DIR --no-check-certificate 'https://drive.google.com/uc?export=download&id=1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy' -O "$EVAL_DIR/eval.zip"
unzip -d $EVAL_DIR ${EVAL_DIR}eval.zip 

# Download VQAv2 (More than 12 GB)
wget -P $EVAL_DIR/vqav2 http://images.cocodataset.org/zips/test2015.zip -O "$EVAL_DIR/vqav2/test2015.zip"
unzip -d $EVAL_DIR/vqav2 ${EVAL_DIR}/vqav2/test2015.zip

# Download GQA (More than 20 GB)
mkdir -p $EVAL_DIR/gqa/data
wget -P $EVAL_DIR/gqa/data  https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip -O "$EVAL_DIR/gqa/data/sceneGraphs.zip"
wget -P $EVAL_DIR/gqa/data  https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip -O "$EVAL_DIR/gqa/data/questions1.2.zip"
wget -P $EVAL_DIR/gqa/data  https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -O "$EVAL_DIR/gqa/data/images.zip"
wget -P $EVAL_DIR/gqa/data  https://nlp.stanford.edu/data/gqa/eval.zip -O "$EVAL_DIR/gqa/data/eval.zip"

unzip -d $EVAL_DIR/gqa/data ${EVAL_DIR}/gqa/data/sceneGraphs.zip
unzip -d $EVAL_DIR/gqa/data ${EVAL_DIR}/gqa/data/questions1.2.zip
unzip -d $EVAL_DIR/gqa/data ${EVAL_DIR}/gqa/data/images.zip
unzip -d $EVAL_DIR/gqa/data ${EVAL_DIR}/gqa/data/eval.zip

# Download VisWiz 
wget -P $EVAL_DIR/vizwiz https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip -O "$EVAL_DIR/vizwiz/Annotations.zip"
wget -P $EVAL_DIR/vizwiz https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip -O "$EVAL_DIR/vizwiz/test.zip"
unzip -d $EVAL_DIR/vizwiz ${EVAL_DIR}/vizwiz/Annotations.zip
unzip -d $EVAL_DIR/vizwiz ${EVAL_DIR}/vizwiz/test.zip


# Download ScienceQA --> image doesn't exist in the git repo
git clone https://github.com/lupantech/ScienceQA
cp -r ScienceQA/data/scienceqa/images $EVAL_DIR/scienceqa
cp ScienceQA/data/scienceqa/pid_splits.json $EVAL_DIR/scienceqa
cp ScienceQA/data/scienceqa/problems.json $EVAL_DIR/scienceqa
rm -rf ScienceQA

# Download TextVQA
wget -P $EVAL_DIR/textvqa https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json -O "$EVAL_DIR/textvqa/TextVQA_0.5.1_val.json"
wget -P $EVAL_DIR/textvqa https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -O "$EVAL_DIR/textvqa/train_val_images.zip"
unzip -d $EVAL_DIR/textvqa ${EVAL_DIR}/textvqa/train_val_images.zip

# Download POPE
wget -P $EVAL_DIR/pope https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco -O "$EVAL_DIR/pope/coco"

# Download MME
# TBD: 
# The benchmark dataset is collected by Xiamen University for academic research only. You can email yongdongluo@stu.xmu.edu.cn to obtain the dataset, according to the following requirement.

# Download MMBench and MMBench-CN
wget -P $EVAL_DIR/mmbench https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv -O "$EVAL_DIR/mmbench/mmbench_dev_20230712.tsv"
wget -P $EVAL_DIR/mmbench https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv -O "$EVAL_DIR/mmbench/mmbench_dev_cn_20231003.tsv"


# Download SEED-Bench
# TBD: 

# Download LLaVA-Bench-in-the-Wild
git clone git@hf.co:datasets/liuhaotian/llava-bench-in-the-wild
cp -r llava-bench-in-the-wild/images $EVAL_DIR/llava-bench-in-the-wild
rm -rf llava-bench-in-the-wild


# Down MM-Vet
wget -P $EVAL_DIR/mm-vet https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip -O "$EVAL_DIR/mm-vet/mm-vet.zip"
unzip -d $EVAL_DIR/mm-vet ${EVAL_DIR}/mm-vet/mm-vet.zip

# Updated on Jun.18th
# Download Evaluation Benchmarks for LLaVA v1.5

# Download Scripts (containing custom annotations, scripts, and the prediction files with LLaVA v1.5)
wget -P $EVAL_DIR --no-check-certificate 'https://drive.google.com/uc?export=download&id=1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy' -O "$EVAL_DIR/eval.zip"
unzip -d $EVAL_DIR ${EVAL_DIR}eval.zip 

# Download VQAv2 (More than 12 GB)
wget -P $EVAL_DIR/vqav2 http://images.cocodataset.org/zips/test2015.zip -O "$EVAL_DIR/vqav2/test2015.zip"
unzip -d $EVAL_DIR/vqav2 ${EVAL_DIR}/vqav2/test2015.zip

# Download GQA (More than 20 GB)
mkdir -p $EVAL_DIR/gqa/data
wget -P $EVAL_DIR/gqa/data  https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip -O "$EVAL_DIR/gqa/data/sceneGraphs.zip"
wget -P $EVAL_DIR/gqa/data  https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip -O "$EVAL_DIR/gqa/data/questions1.2.zip"
wget -P $EVAL_DIR/gqa/data  https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip -O "$EVAL_DIR/gqa/data/images.zip"
wget -P $EVAL_DIR/gqa/data  https://nlp.stanford.edu/data/gqa/eval.zip -O "$EVAL_DIR/gqa/data/eval.zip"

unzip -d $EVAL_DIR/gqa/data ${EVAL_DIR}/gqa/data/sceneGraphs.zip
unzip -d $EVAL_DIR/gqa/data ${EVAL_DIR}/gqa/data/questions1.2.zip
unzip -d $EVAL_DIR/gqa/data ${EVAL_DIR}/gqa/data/images.zip
unzip -d $EVAL_DIR/gqa/data ${EVAL_DIR}/gqa/data/eval.zip

# Download VisWiz 
wget -P $EVAL_DIR/vizwiz https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip -O "$EVAL_DIR/vizwiz/Annotations.zip"
wget -P $EVAL_DIR/vizwiz https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip -O "$EVAL_DIR/vizwiz/test.zip"
unzip -d $EVAL_DIR/vizwiz ${EVAL_DIR}/vizwiz/Annotations.zip
unzip -d $EVAL_DIR/vizwiz ${EVAL_DIR}/vizwiz/test.zip


# Download ScienceQA --> image doesn't exist in the git repo
git clone https://github.com/lupantech/ScienceQA
cp -r ScienceQA/data/scienceqa/images $EVAL_DIR/scienceqa
cp ScienceQA/data/scienceqa/pid_splits.json $EVAL_DIR/scienceqa
cp ScienceQA/data/scienceqa/problems.json $EVAL_DIR/scienceqa
rm -rf ScienceQA

# Download TextVQA
wget -P $EVAL_DIR/textvqa https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json -O "$EVAL_DIR/textvqa/TextVQA_0.5.1_val.json"
wget -P $EVAL_DIR/textvqa https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -O "$EVAL_DIR/textvqa/train_val_images.zip"
unzip -d $EVAL_DIR/textvqa ${EVAL_DIR}/textvqa/train_val_images.zip

# Download POPE
wget -P $EVAL_DIR/pope https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco -O "$EVAL_DIR/pope/coco"

# Download MME
# TBD: 
# The benchmark dataset is collected by Xiamen University for academic research only. You can email yongdongluo@stu.xmu.edu.cn to obtain the dataset, according to the following requirement.

# Download MMBench and MMBench-CN
wget -P $EVAL_DIR/mmbench https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv -O "$EVAL_DIR/mmbench/mmbench_dev_20230712.tsv"
wget -P $EVAL_DIR/mmbench https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv -O "$EVAL_DIR/mmbench/mmbench_dev_cn_20231003.tsv"


# Download SEED-Bench
# TBD: 

# Download LLaVA-Bench-in-the-Wild
git clone git@hf.co:datasets/liuhaotian/llava-bench-in-the-wild
cp -r llava-bench-in-the-wild/images $EVAL_DIR/llava-bench-in-the-wild
rm -rf llava-bench-in-the-wild


# Down MM-Vet
wget -P $EVAL_DIR/mm-vet https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip -O "$EVAL_DIR/mm-vet/mm-vet.zip"
unzip -d $EVAL_DIR/mm-vet ${EVAL_DIR}/mm-vet/mm-vet.zip