'''
Simple diagnosis script to make sure LLava runs in 
the current environment
'''

from scripts.model_util import Llava
from PIL import Image
import requests

model = Llava()

prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

preds = model.generate(prompt, image, max_length=30)
print(preds)