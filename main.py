import shutil
from fastapi import FastAPI, File, HTTPException, UploadFile, Form, Request, Depends, status
from fastapi.responses import FileResponse, RedirectResponse

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
              "a photo of a document", "a photo of an animal","a photo of animals", "a photo of people in landscape","a photo of nature"]

class_dict = {"a photo of a food":"food","a photo of a landscape":"nature", "a photo of a person":"human","a photo of people":"human", 
              "a photo of a document":"docs", "a photo of an animal":"animal","a photo of animals":"animal","a photo of people in landscape":"human","a photo of nature":"nature","others":"others"}



abs_path = os.path.dirname(os.path.realpath(__file__))
templates = Jinja2Templates(directory=f"{abs_path}/templates")
templates.env.filters["b64encode"] = lambda data: data.encode("utf-8").decode("utf-8")

# FastAPI() 객체 생성
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

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

@app.post("/upload")
async def predict_api(request: Request,image_files: List[UploadFile] = File(...),db: Session = Depends(get_db)):
    for image_file in image_files:
        image_data = await image_file.read()
        filename = image_file.filename
        categoreyes = models.Categoreyes(data=image_data, filename=filename)
        db.add(categoreyes)
    db.commit()
    
    return RedirectResponse(url=app.url_path_for("show_image"), status_code=status.HTTP_303_SEE_OTHER)

@app.get("/show_image")
async def show_image(request: Request,db: Session = Depends(get_db)):
    categoreyes_list = db.query(models.Categoreyes).all()
    show_list = []
    for img in categoreyes_list:
        image_data = base64.b64encode(img.data).decode('utf-8')
        show_list.append(image_data)
    return templates.TemplateResponse("show_image.html",
                                      {"request": request,"show_list":show_list})    

@app.get("/predict")
async def predict(request: Request, db: Session = Depends(get_db)):
    categoreyes_list = db.query(models.Categoreyes).all()
    predictions = []
    classified = {"human":[],"nature":[],"food":[],"docs":[],"animal":[],"others":[]}
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
        image_data = base64.b64encode(categoreyes.data).decode('utf-8')
        

        if max(probs.tolist()[0])>0.45:
            result = class_dict[class_name[probs.argmax()]]
            if result =="human": db.add(models.Human(data=categoreyes.data, filename=categoreyes.filename))
            elif result == "nature": db.add(models.Nature(data=categoreyes.data, filename=categoreyes.filename))
            elif result == "food": db.add(models.Food(data=categoreyes.data, filename=categoreyes.filename))
            elif result == "docs": db.add(models.Docs(data=categoreyes.data, filename=categoreyes.filename))
            elif result == "animal": db.add(models.Animal(data=categoreyes.data, filename=categoreyes.filename))

        else:
            result = "others"
            db.add(models.Others(data=categoreyes.data, filename=categoreyes.filename))
        
        predictions.append({"filename": categoreyes.filename, "result": result,"img":image_data})
        classified[result].append(image_data)

    db.commit()


    #해당 카테고리에 파일이 없다면 폴더 보이지 않도록 삭제
    delete_keys = []
    for classs_key in classified.keys():
        if len(classified[classs_key])==0:
            delete_keys.append(classs_key)
    for k in delete_keys:      
        classified.pop(k, None)

    return templates.TemplateResponse("predicted.html",
                                      {"request": request,
                                       "predictions": predictions,
                                       "classified":classified})    

@app.get("/animal_images")
async def animal_images(request: Request, db: Session = Depends(get_db)):
    animal_list = db.query(models.Animal).all()
    show_list = []
    for img in animal_list:
        image_data = base64.b64encode(img.data).decode('utf-8')
        show_list.append(image_data)
    return templates.TemplateResponse("category_show.html",
                                      {"request": request,"show_list":show_list,"category":"animal"})    


@app.get("/human_images")
async def human_images(request: Request, db: Session = Depends(get_db)):
    animal_list = db.query(models.Human).all()
    show_list = []
    for img in animal_list:
        image_data = base64.b64encode(img.data).decode('utf-8')
        show_list.append(image_data)
    return templates.TemplateResponse("category_show.html",
                                      {"request": request,"show_list":show_list,"category":"human"}) 

@app.get("/food_images")
async def food_images(request: Request, db: Session = Depends(get_db)):
    food_list = db.query(models.Food).all()
    show_list = []
    for img in food_list:
        image_data = base64.b64encode(img.data).decode('utf-8')
        show_list.append(image_data)
    return templates.TemplateResponse("category_show.html",
                                      {"request": request,"show_list":show_list,"category":"food"}) 

@app.get("/docs_images")
async def docs_images(request: Request, db: Session = Depends(get_db)):
    docs_list = db.query(models.Docs).all()
    show_list = []
    for img in docs_list:
        image_data = base64.b64encode(img.data).decode('utf-8')
        show_list.append(image_data)
    return templates.TemplateResponse("category_show.html",
                                      {"request": request,"show_list":show_list,"category":"docs"}) 

@app.get("/nature_images")
async def nature_images(request: Request, db: Session = Depends(get_db)):
    nature_list = db.query(models.Nature).all()
    show_list = []
    for img in nature_list:
        image_data = base64.b64encode(img.data).decode('utf-8')
        show_list.append(image_data)
    return templates.TemplateResponse("category_show.html",
                                      {"request": request,"show_list":show_list,"category":"nature"}) 

@app.get("/others_images")
async def others_images(request: Request, db: Session = Depends(get_db)):
    others_list = db.query(models.Others).all()
    show_list = []
    for img in others_list:
        image_data = base64.b64encode(img.data).decode('utf-8')
        show_list.append(image_data)
    return templates.TemplateResponse("category_show.html",
                                      {"request": request,"show_list":show_list,"category":"others"}) 

@app.get("/save_images")
async def save_images(request: Request, db: Session = Depends(get_db)):
    category = request.query_params.get('category', None)
    if category == "food_images": save_category = db.query(models.Food).all()
    elif category == "haman_images": save_category = db.query(models.Human).all()
    elif category == "nature_images": save_category = db.query(models.Nature).all()
    elif category == "docs_images": save_category = db.query(models.Docs).all()
    elif category == "animal_images": save_category = db.query(models.Animal).all()
    elif category == "others_images": save_category = db.query(models.Others).all()


    save_folder = category
    os.makedirs(save_folder, exist_ok=True)

    for item in save_category:
        bin_img = item.data
        filename = item.filename
        # Assuming the 'file_path' field contains the path to the image file
        image_path = os.path.join(save_folder, filename)

        # Save the binary data to the file
        with open(image_path, "wb") as img_file:
            img_file.write(bin_img)

    # Create a zip file containing all the downloaded images
    shutil.make_archive(save_folder, 'zip', save_folder)

    # Provide the zip file for download
    return FileResponse(f"{save_folder}.zip", filename=f"{save_folder}.zip", media_type="application/zip")