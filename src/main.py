from fastapi import FastAPI, UploadFile, File, Depends
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
import pytesseract
import numpy as np
from io import BytesIO
import cv2
import base64
import layoutparser as lp

app = FastAPI()

class Params(BaseModel):
    lang: Optional[str] = "eng"
    feature_type: Optional[str] = "LINE"

@app.get("/")
def home():
    return "OCR Layout Parser with FastAPI - Version 1.0"

@app.post("/ocr/")
async def submit(params: Params = Depends(), files: List[UploadFile] = File(...)):
    results = {}

    # Tesseract OCR Agent
    ocr_agent = lp.TesseractAgent(languages=params.lang)

    for file in files:
        # Read the image file as bytes
        img_data = await file.read()

        # Use cv2.imdecode to convert the image data to a numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert the image to grayscale (1 channel)
        if len(img_cv2.shape) >= 3:
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)

        # Up-sample
        img_cv2 = cv2.resize(img_cv2, (0, 0), fx=2, fy=2)

        # https://stackoverflow.com/questions/71289347/pytesseract-improving-ocr-accuracy-for-blurred-numbers-on-an-image
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img_cv2 = cv2.filter2D(img_cv2, -1, sharpen_kernel)
        img_cv2 = cv2.threshold(img_cv2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Down-sample
        img_cv2 = cv2.resize(img_cv2, (0, 0), fx=0.5, fy=0.5)

        # Run Layout Parser with Tesseract
        res = ocr_agent.detect(img_cv2, return_response=True)

        if params.feature_type == "PAGE":
            agg_level = lp.TesseractFeatureType.PAGE
        elif params.feature_type == "BLOCK":
            agg_level = lp.TesseractFeatureType.BLOCK
        elif params.feature_type == "PARA":
            agg_level = lp.TesseractFeatureType.PARA
        elif params.feature_type == "LINE":
            agg_level = lp.TesseractFeatureType.LINE
        elif params.feature_type == "WORD":
            agg_level = lp.TesseractFeatureType.WORD

        layout_tesseract  = ocr_agent.gather_data(res, agg_level=agg_level)

        # Create dict with id and text
        layout_tesseract_text_id = {}
        full_text = ""
        for index, text in enumerate(layout_tesseract.get_texts()):
            layout_tesseract_text_id[layout_tesseract.get_info('id')[index]] = text
            full_text += text + "\n"
        
        """
        # Draw text of detected layout 
        layout_tesseract_image = lp.draw_box(img_cv2, layout_tesseract, box_width=3, show_element_id=True)
        with BytesIO() as output_buffer:
            layout_tesseract_image.save(output_buffer, format="PNG")
            image_data = output_buffer.getvalue()
        final_image_base64 = base64.b64encode(image_data)
        """
        _, img_bytes = cv2.imencode('.jpg', img_cv2)
        final_image_base64 = base64.b64encode(img_bytes).decode('utf-8')

        results[file.filename] = {}
        results[file.filename]['ocr'] = full_text
        results[file.filename]['final_image_base64'] = final_image_base64

    return {"results": results,
            "params": params}