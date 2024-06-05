'''
Simple diagnosis script to make sure BLIP2 runs in 
the current environment
'''

from src.model_util import OFlamingo
from PIL import Image
import requests

model = OFlamingo()

prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

preds = model.generate(prompt, image, max_length=30)
print(preds)