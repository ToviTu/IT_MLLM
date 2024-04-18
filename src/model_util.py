import torch
import os
import glob
import re
from torch import nn
from transformers import BitsAndBytesConfig
from transformers import pipeline
from transformers import LlavaForConditionalGeneration
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from open_flamingo import create_model_and_transforms
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, LlamaForCausalLM
import subprocess


HF_TOKEN = "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"

class OFlamingo(nn.Module):

    '''
    A wrapper for the OpenFlamingo Model; This class currently does not work 
    due to the recent update of HuggingFace; Downgrading transformers to 4.28.1
    solves the problem; However, Llava requires transformers>=4.35.1
    '''

    def __init__(self, cache_dir="/scratch/t.tovi/models/"):
        super().__init__()
        model_id = "openflamingo/OpenFlamingo-9B-vitl-mpt7b"
        cache_dir = cache_dir if cache_dir[-1] == '/' else cache_dir + "/"

        # Default to gpu
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Initialize
        self.text_model, self.vision_model, self.processor = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="mosaicml/mpt-7b",
            tokenizer_path="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=4,
        )

        # Load pretrained weights
        checkpoint_path = hf_hub_download(
            model_id, 
            "checkpoint.pt",
            local_dir=cache_dir+model_id,
            cache_dir=cache_dir+model_id,
            local_dir_use_symlinks=False,
            token=HF_TOKEN
        )
        self.text_model.load_state_dict(torch.load(checkpoint_path), strict=False)

        self.text_model.to(self.device)
        

    def preprocess(self, text, images):
        if type(images) != "<class 'list'>":
            images = [images]

        # Prepare images
        vision_x = [self.vision_model(img).unsqueeze(0) for img in images]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0).to(self.device)

        # Prepare texts
        self.processor.padding_side = "left" 
        lang_x = self.processor(
            [text],
            return_tensors="pt",
        ).to(self.device)

        return vision_x, lang_x

    def forward(self, text, images):

        vision_x, lang_x = self.preprocess(text, images)
        return self.text_model(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"]
        )
    
    def generate(self, text, images, **gargs):

        vision_x, lang_x = self.preprocess(text, images)
        preds = self.text_model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"],
            attention_mask=lang_x["attention_mask"],
            **gargs
        )

class Llava(nn.Module):

    '''
    A wrapper for the pretrained model; It should handle
    batched inputs including images and texts
    '''

    def __init__(self, cache_dir='/scratch/t.tovi/models/', quantization="False", model_id="llava-hf/llava-1.5-7b-hf"):
        super().__init__()

        # Default to gpu
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # Support for quantized models
        if quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
        elif quantization == 'bfloat16':
            self.precision = torch.bfloat16
        else:
            self.precision = torch.float32
        print(f'Setting precision to {self.precision}')
        
        # Load pretrained weights
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir = cache_dir,
            torch_dtype = self.precision
        ).to(self.device)

        # Load autoprocessor
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir = cache_dir
        )

        print('--------------Finished Loading Model--------------')
        output = subprocess.check_output(
            ["nvidia-smi", 
            "--query-gpu=memory.total,memory.used,memory.free", 
            "--format=csv,nounits,noheader"], 
            encoding='utf-8'
        )

        # Print the output
        print(output)

    def forward(self, text, image=None):

        # Encode inputs
        inputs = self.processor(text=text, images=image, return_tensors='pt').to(self.device, self.precision)

        # Forward
        return self.model(**inputs)
    
    def generate(self, text, image=None, **gargs):

        with torch.no_grad():

            # Encode inputs
            inputs = self.processor(text=text, images=image, return_tensors='pt')

            # Move to the proper device
            inputs = {
                'input_ids': inputs['input_ids'].to(self.device),
                'attention_mask': inputs['attention_mask'].to(self.device),
                'pixel_values': inputs['pixel_values'].to(self.device, self.precision) \
                                if inputs['pixel_values'] is not None else None,
            }

            # Generate
            preds = self.model.generate(**inputs, **gargs)

        # Decode
        return self.processor.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
class Blip2(nn.Module):

    '''
    A wrapper for the pretrained model; It should handle
    batched inputs including images and texts
    '''

    def __init__(self, cache_dir='/scratch/t.tovi/models/', quantization="False", model_id="Salesforce/blip2-opt-2.7b"):
        super().__init__()

        # Default to gpu
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Load pretrained weights
        self.model =  Blip2ForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir = cache_dir
        ).to(self.device)

        # Load autoprocessor
        self.processor = Blip2Processor.from_pretrained(
            model_id,
            cache_dir = cache_dir
        )

    def forward(self, text, image):

        # Encode inputs
        inputs = self.processor(text=text, images=image, return_tensors='pt').to(self.device)

        # Forward
        return self.model(**inputs)
    
    def generate(self, text, image, **gargs):

        with torch.no_grad():

            # Encode inputs
            inputs = self.processor(text=text, images=image, return_tensors='pt').to(self.device)

            # Generate
            preds = self.model.generate(**inputs, **gargs)

        # Decode
        return self.processor.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)

class Yi(nn.Module):

    '''
    A wrapper for the pretrained model
    '''

    def __init__(self, cache_dir='/scratch/t.tovi/models/', model_id="01-ai/Yi-34B-Chat"):
        super().__init__()

        # Default to gpu
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.precision = torch.bfloat16
        
        # Load pretrained weights
        self.model =  AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir = cache_dir,
            device_map=self.device,
            torch_dtype='auto'
        )

        # Load autoprocessor
        self.processor = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir = cache_dir
        )

    def forward(self, text):

        # Encode inputs
        inputs = self.processor(text=text, return_tensors='pt').to(self.device)

        # Forward
        return self.model(**inputs)
    
    def generate(self, text, **gargs):

        with torch.no_grad():

            # Encode inputs
            inputs = self.processor(text=text, return_tensors='pt', padding=True).to(self.device)

            # Generate
            preds = self.model.generate(**inputs, **gargs)

        # Decode
        return self.processor.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)

class Llama2(nn.Module):

    '''
    A wrapper for the pretrained model
    '''

    def __init__(self, cache_dir='/scratch/t.tovi/models/', model_id="meta-llama/Llama-2-7b-chat-hf"):
        super().__init__()

        # Default to gpu
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.precision = torch.bfloat16
        
        # Load pretrained weights
        self.model =  LlamaForCausalLM.from_pretrained(
            model_id,
            cache_dir = cache_dir,
            token=HF_TOKEN,
        ).to(self.device, self.precision)

        # Load autoprocessor
        self.processor = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir = cache_dir,
            token=HF_TOKEN,
        )
        self.processor.pad_token = "[PAD]"
        self.processor.padding_side = "left"

    def forward(self, text):

        # Encode inputs
        inputs = self.processor(text=text, return_tensors='pt').to(self.device)

        # Forward
        return self.model(**inputs)
    
    def generate(self, text, **gargs):

        with torch.no_grad():

            # Encode inputs
            inputs = self.processor(text=text, return_tensors='pt').to(self.device)

            # Generate
            preds = self.model.generate(**inputs, **gargs)

        # Decode
        return self.processor.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=False)