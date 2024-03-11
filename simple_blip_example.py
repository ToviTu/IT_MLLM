'''
Simple diagnosis script to make sure OpenFlamingo runs in 
the current environment
'''

from scripts.model_util import Blip2
from PIL import Image
import requests

model = Blip2()

prompt = "USER: What's the content of the image?\nASSISTANT:"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"
image = Image.open(requests.get(url, stream=True).raw)

preds = model.generate(prompt, image, max_new_tokens=30)
print(preds)