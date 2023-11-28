import torch
from PIL import Image
import requests
from transformers import Blip2Processor, Blip2Model

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img_name = "images/person_1.jpg"
image = Image.open(img_name)
inputs = processor(images=image, return_tensors="pt")
qformer_outputs = model.get_qformer_features(**inputs)
print(qformer_outputs)