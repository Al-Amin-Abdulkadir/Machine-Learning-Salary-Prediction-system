from fastapi import FastAPI, Request
import joblib
import numpy as np
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from config.config import TEMPLATES_DIR


app = FastAPI(title="Salary Predictor")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
model = joblib.load("model.pkl")

@app.get("/home", response_class=HTMLResponse)
def home(request : Request):
    return templates.TemplateResponse(
        "home.html",
        {
            "request" : request,
            
        }
    )