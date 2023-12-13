import numpy as np
import pandas as pd
import re
import openai
from diffusers import StableDiffusionPipeline, DDIMScheduler
from datasets import load_dataset

### timeout fn
# from https://stackoverflow.com/questions/21827874/timeout-a-function-windows
from threading import Thread
import functools

def timeout(timeout):
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = [Exception('function [%s] timeout [%s seconds] exceeded!' % (func.__name__, timeout))]
            def newFunc():
                try:
                    res[0] = func(*args, **kwargs)
                except Exception as e:
                    res[0] = e
            t = Thread(target=newFunc)
            t.daemon = True
            try:
                t.start()
                t.join(timeout)
            except Exception as je:
                print ('error starting thread')
                raise je
            ret = res[0]
            if isinstance(ret, BaseException):
                raise ret
            return ret
        return wrapper
    return deco

### setup
# dataset
dataset = load_dataset("squad", split="train")

# stable diffusion
scheduler =  DDIMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    scheduler=scheduler,
).to("cuda")

# OpenAI api
# Set your API key here
api_key = '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'

# Initialize OpenAI's API client
client = openai.OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=api_key,
)

# prompts
negative_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"
# TODO: change to no in-context examples to speed up api response and reduce cost
# tasks
img_gen_task = "Select the full relevant quote from the context that allows the answer to be derived (it must contain the answer) and summarize the information relevant to the question and answer in an extremely short single image description that focuses on important expressive visual features with no mention of representation, meaning, or reason. Lastly, provide an extremely concise image feature list that only lists the shorthand names and numbers relevant to the image without description.  Respond in the format: \"Quote: <full_sentence_quote>\nImage Description: An image showing <image_description>\nImage Feature List: Tags: <image_feature_list>\"\n"
rationale_task = "Answer the question using only the text from the context and the image and explain your rationale. First provide the rationale, then the answer, all in 1 paragraph. Use only the provided information and interpret the image  as showing more information than is described.  Be as concise as possible while referencing the text and image and only address information that directly contributes to the answer. You must use both the context and the image in the rationale. Start by rephrasing \"In the [text/image]\".\n"

# prompts
img_gen_prompt = "Context: {}\nQuestion: {}\nAnswer: {}\nTask: " + img_gen_task
rationale_prompt = "[Image: {} {}]\n[Context: {}]\n[Question: {}]\n[Answer: {}]\nTask: " + rationale_task

# output data
header = ['context','question','answer','split_1','quote','split_2','image_description','image_text','image_name','rationale']
# init csv
df = pd.DataFrame([], columns=header)
file_name = "all_examples.csv"
df.to_csv(file_name, index=False) #, sep='\t')

### run methods
@timeout(60)
def run_prompt(prompt):
    return client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-3.5-turbo",
    )

# run for one example
def run_once(data, file_name="all_examples.csv", gen_steps=50, MAX_TRIES=2):
    # prompt for image descript generation
    prompt_img = img_gen_prompt.format(data['context'], data['question'], data['answers']['text'][0])
    for i in range(MAX_TRIES):
        try:
            is_api=True
            chat_completion_img = run_prompt(prompt_img)
            is_api=False
            parse = re.findall(r"^\W*Quote: \"(.+)\"\W+Image Description: (.+)\W+Image Feature List: (.+)\W*$", chat_completion_img.choices[0].message.content)[0]
            quote = parse[0]
            img_descr = parse[1]
            img_text = parse[2][6:]
            img = img_descr + ' ' + img_text #.rstrip('.')
            split_1 = '' #splits[0]
            split_2 = '' #splits[1]
            break
        except Exception as e:
            #print(e)
            if i==MAX_TRIES-1:
                print("Prompt 1 (img) timed out")
                return None

    # prompt for rationale generation
    prompt_rat = rationale_prompt.format(img_descr, img_text, data['context'], data['question'], data['answers']['text'][0])
    for i in range(MAX_TRIES):
        try:
            chat_completion_rat = run_prompt(prompt_rat)
            rat_response = re.findall(r"^(.+)\n?", chat_completion_rat.choices[0].message.content)[0]
            #print(rat_response)
            break
        except Exception as e:
            # print(e)
            # print(prompt_rat)
            if i==MAX_TRIES-1:
                print("Prompt 2 (rationale) timed out")
                return None
    # print()
    
    #img gen
    image = pipe(img_descr, negative_prompt=negative_prompt, num_inference_steps=gen_steps).images[0]
    image.save("images/{}.jpg".format(data['id']))
    
    #csv
    to_out = [[
        data['context'], data['question'], data['answers']['text'][0],
        split_1, quote, split_2,
        img_descr, img_text,
        "{}.jpg".format(data['id']),
        rat_response,
    ]]
    
    # create dataframe and append
    df = pd.DataFrame(to_out, columns=header)
    df.to_csv(file_name, mode='a', index=False, header=False) #, sep='\t')
    
    return to_out

### run
num_to_gen = 2000
increment_num = 10
start_num = 0

start_ind = start_num*increment_num
break_at = num_to_gen*increment_num + start_ind
for i in range(start_ind,dataset.num_rows,increment_num):
    if i >= break_at: break
    data = dataset[i]
    print('{}, '.format(i))
    to_out = run_once(data, gen_steps=50, MAX_TRIES=4)
    if to_out is None:
        continue
    else:
        pass

print('done')
