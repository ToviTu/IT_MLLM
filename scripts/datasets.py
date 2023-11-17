from json import load
from re import split
import re
from datasets import load_dataset
import os
import random
import torch
import requests
from PIL import Image

open_image_url = lambda url: Image.open(requests.get(url, stream=True).raw)


class Base_dataset:
    def __init__(self, precision=torch.bfloat16, device=0) -> None:
        self.precision = precision
        self.device = device

        placeholder_image_url = "https://www.ledr.com/colours/white.jpg"
        self.palceholder_image = open_image_url(placeholder_image_url)

        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def postprocess_vqa_generation(self, predictions):
        answer = re.split("Question|Answer|Short", predictions, 1)[0]
        answer = re.split(", ", answer, 1)[0]
        return answer

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText

    def normalize(self, text):
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        text = text.strip()
        text = self.processPunctuation(text)
        text = self.processDigitArticle(text)

    def image_preprocess_batch(self, image_processor, images: list) -> torch.Tensor:
        vision_x = [image_processor(image).unsqueeze(0) for image in images]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        return vision_x

    def text_processor_factory(self, tokenizer, padding_side):
        tokenizer.padding_side = padding_side
        return lambda text: tokenizer([text], return_tensors="pt")


DATASET_DIR = (
    "/mnt/d/datasets/"
    if os.environ.get("DATASET_DIR") == None
    else os.environ["DATASET_DIR"]
)


class VQA_dataset(Base_dataset):
    def __init__(self, precision=torch.bfloat16, device=0) -> None:
        super().__init__(precision=precision, device=device)
        self.train_dataset = load_dataset(
            "Graphcore/vqa", split="train", cache_dir=DATASET_DIR
        )
        self.val_dataset = load_dataset("Graphcore/vqa", split="validation", cache_dir=DATASET_DIR)  # type: ignore

    def qa_prompt(self, question, answer=None) -> str:
        return f"<image>Question:{question} Short answer:{answer if answer is not None else ''}{'<|endofchunk|>' if answer is not None else ''}"

    def make_in_context(self, d, shots=4):
        d = self.train_dataset
        images = []
        prompt = ""
        for _ in range(shots):
            idx = int(random.random() * len(d))
            image = d[idx]["image"]
            question = d[idx]["question"]
            answers = d[idx]["answers"][0]

            images.append(image)
            prompt += self.qa_prompt(question, answers) + "\n"
            if shots == 0:
                prompt = prompt.replace("<image>", "")
        return images, prompt

    def eval(
        self, model, image_encoder, text_encoder, shots=4, batch_size=8, early_stop=2000
    ):
        predictions = []
        answers = []

        for idx in range(0, len(self.val_dataset), batch_size):
            d_clip = self.val_dataset[idx * batch_size : (idx + 1) * batch_size]
            questions = d_clip["question"]
