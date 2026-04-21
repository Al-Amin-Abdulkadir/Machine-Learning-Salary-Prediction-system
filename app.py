from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Salary Predictor")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

model = joblib.load(str(BASE_DIR / "model.pkl"))



@app.get("/home", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse("prediction_page.html", {"request": request})


@app.post("/predict")
def predict(experience: float, age: float):
    data = np.array([[experience, age]])
    prediction = model.predict(data)
    return {"predicted_salary": round(float(prediction[0]), 2)}
