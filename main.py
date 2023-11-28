from fastapi import FastAPI, File, UploadFile, Form, Request, Depends, status
from fastapi.responses import RedirectResponse

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import time
from transformers import CLIPProcessor, CLIPModel
import os
import io
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import List
import asyncio

from database import engine, SessionLocal 
from sqlalchemy.orm import Session
import base64  # 추가된 부분

import models

#Define model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# models에 정의한 모든 클래스, 연결한 DB엔진에 테이블로 생성
models.Base.metadata.create_all(bind=engine)

#Define categories
class_name = ["a photo of a food","a photo of a landscape", "a photo of a person","a photo of people", 
              "a photo of a document", "a photo of an animal", "a photo of people in landscape","a photo of nature"]

class_dict = {"a photo of a food":"food","a photo of a landscape":"landscape", "a photo of a person":"human","a photo of people":"human", 
              "a photo of a document":"docs", "a photo of an animal":"animal","a photo of people in landscape":"human","a photo of nature":"nature"}





abs_path = os.path.dirname(os.path.realpath(__file__))
templates = Jinja2Templates(directory=f"{abs_path}/templates")
templates.env.filters["b64encode"] = lambda data: data.encode("utf-8").decode("utf-8")

# FastAPI() 객체 생성
app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        # 마지막에 무조건 닫음
        db.close()

@app.get("/",response_class="HTMLResponse")
async def home(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.post("/send_image")
async def predict_api(request: Request,image_files: List[UploadFile] = File(...),db: Session = Depends(get_db)):
    for image_file in image_files:
        image_data = await image_file.read()
        filename = image_file.filename
        categoreyes = models.Categoreyes(data=image_data, filename=filename)
        
        db.add(categoreyes)
        
    db.commit()
    return RedirectResponse(url=app.url_path_for("predict"), status_code=status.HTTP_303_SEE_OTHER)

@app.get("/predict")
async def predict(request: Request, db: Session = Depends(get_db)):
    categoreyes_list = db.query(models.Categoreyes).all()
    predictions = []
    classified = {"human":[],"landscape":[],"food":[],"docs":[],"animal":[],"others":[]}
    for idx in range(len(categoreyes_list)):
        categoreyes=categoreyes_list[idx]
        # 1. make buffer from bytes
        buffer = io.BytesIO(categoreyes.data)

        # 2. decode image form buffer
        image = Image.open(buffer)
        
        inputs = processor(text=class_name, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

        result = class_dict[class_name[probs.argmax()]]
        image_data = base64.b64encode(categoreyes.data).decode('utf-8')

        predictions.append({"filename": categoreyes.filename, "result": result,"img":image_data})
        classified[result].append(idx)

    return templates.TemplateResponse("predicted.html",
                                      {"request": request,
                                       "predictions": predictions,
                                       "classified":classified})    


