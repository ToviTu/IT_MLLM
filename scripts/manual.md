# Documentation for provided scripts

This page is dedicated to guide the usage of the provided tools for this project.

## Default Cache Dir

This project commonly uses pretrained checkpoints downloaded form the HuggingFace platform. By default, the weights are saved in the "/scratch/t.tovi/models/" directory on the "Chenguang03" node due to the driver error on "Chenguang01" and little available space on "Chenguang02".

**Update**

Run the ". scripts/set_environ_var.sh" to properly set all default envrionment variables before running any huggingface codes.