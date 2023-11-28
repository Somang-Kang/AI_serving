from PIL import Image
import time
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
img_name = "images/person_1.jpg"
image = Image.open(img_name)
class_name = ["a photo of a food", "a photo of a landscape", "a photo of a person", "a photo of a document", "a photo of a animal", "a photo of people in landscape"]
start = time.time()
inputs = processor(text=class_name, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
end = time.time()
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

print(probs)
print(class_name[probs.argmax()])
print(f"{end - start:.5f} sec")
