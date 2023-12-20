# Instruction-tuned-Flamingo-MLLM

Final project for CSE527A Natural Language Processing at Washington University in Saint Louis. 

## Abstract

Since the OpenFlamingo family of models provides various versions of the smallest 3B VLM, we evaluate 2 example models to understand their behavior and choose the most appropriate one as the baseline model. We also evaluate the LLM backbone, Mpt-base-1B, to test whether their architecture is susceptible to catastrophic forgetting. The 3 models are OpenFlamingo-base-3B, the instruction-tuned OpenFlamingo-dolly-3B, and the backbone Mpt-base-1B. Since the LLM backbone does not accept visual inputs, its performance on VQAv2 is not evaluated. As a result, the performance of the three models is on par with each other, with the OpenFlmingo-base-3B achieving the highest score.


<img width="812" alt="pipeline" src="https://github.com/ToviTu/Instruction-tuned-Flamingo-MLLM/assets/52998198/b6ec2cc4-14bf-49dc-b45b-60e5f2c447bb">
