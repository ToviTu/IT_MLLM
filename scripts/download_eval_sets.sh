#! /bin/bash

# VQAv2 evaluation set
wget -P ${HF_DATASETS_CACHE} http://images.cocodataset.org/zips/test2015.zip
unzip -d ${HF_DATASETS_CACHE} ${HF_DATASETS_CACHE}test_2015.zip
rm ${HF_DATASETS_CACHE}test_2015.zip