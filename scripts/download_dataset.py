import os
import requests
import zipfile

DATASET_DIR = (
    "/mnt/d/datasets/"
    if os.environ.get("DATASET_DIR") == None
    else os.environ["DATASET_DIR"]
)


DATASET_URL = {
    "vqa_val_anno": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
    "vqa_val_quest": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "vqa_val_image": "http://images.cocodataset.org/zips/val2014.zip",
    "vqa_train_anno": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "vqa_train_quest": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "train2014": "http://images.cocodataset.org/zips/train2014.zip",
    "trivia_qa": "https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz",
}


def unzip_dataset(key, dir):
    try:
        print(f"Unzipping file {dir + key}")
        with zipfile.ZipFile(dir + key + ".zip", "r") as zip_ref:
            zip_ref.extractall(dir)
    except zipfile.BadZipFile:
        print("The file is not a zip file or it is corrupted.")
    except FileNotFoundError:
        print("The ZIP file was not found.")
    except PermissionError:
        print("Permission denied: unable to extract files.")


def get_dataset(key, dir):
    if not os.path.exists(dir + key + ".zip"):
        response = requests.get(val)
        if response.status_code == 200:
            print(f"Downloading dataset: {key}")
            with open(dir + key + ".zip", "wb") as f:
                f.write(response.content)


if __name__ == "__main__":
    for key, val in DATASET_URL.items():
        get_dataset(key, DATASET_DIR)
        unzip_dataset(key, DATASET_DIR)
        if os.path.exists(DATASET_DIR + key + ".zip"):
            os.remove(DATASET_DIR + key + ".zip")
