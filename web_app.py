from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, model_validator
import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
from catboost import CatBoostRegressor
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Dict, Any
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Убираем предупреждения
warnings.filterwarnings('ignore')

MAX_PRICE = 35_000_000

# Добавляем путь к src/features для импорта preprocess_one_flat
sys.path.append(os.path.join(os.path.dirname(__file__), 'src/features'))

# Импортируем наш процессор
from preprocess_one_flat import SingleFlatProcessor


class FlatInput(BaseModel):
    """Модель для валидации входных данных квартиры"""
    комнат: float = Field(..., ge=0.5, le=10)
    площадь_общая: float = Field(..., gt=10, le=500)
    площадь_кухни: float = Field(..., ge=0, le=100)
    площадь_жилая: float = Field(..., ge=0, le=200)
    этаж: int = Field(..., ge=0, le=100)
    этажность_дома: int = Field(..., ge=1, le=100)
    высота_потолков: float = Field(..., ge=1.5, le=5)
    год_постройки: int = Field(..., ge=1800, le=2024)
    тип_дома: str = Field(...)
    время_до_ближайшего_мин: float = Field(..., ge=0, le=180)
    все_метро: str = Field(...)
    лифт_пасс: int = Field(..., ge=0, le=10)
    лифт_груз: int = Field(..., ge=0, le=5)
    ремонт: str = Field(...)
    вид_сделки: int = Field(..., ge=0, le=1)
    is_apartment: int = Field(..., ge=0, le=1)
    студия: int = Field(..., ge=0, le=1)
    
    кухня: int = Field(0, ge=0, le=1)
    спальные_места: int = Field(0, ge=0, le=1)
    шкаф_хранение: int = Field(0, ge=0, le=1)
    техника: int = Field(0, ge=0, le=1)
    диван_кровать: int = Field(0, ge=0, le=1)
    изолированные_комнаты: int = Field(0, ge=0, le=1)
    смежные_комнаты: int = Field(0, ge=0, le=1)
    совмещенный_санузел: int = Field(0, ge=0, le=1)
    раздельный_санузел: int = Field(0, ge=0, le=1)
    
    окна_во_двор: int = Field(0, ge=0, le=1)
    окна_на_улицу: int = Field(0, ge=0, le=1)
    окна_на_солнечную_сторону: int = Field(0, ge=0, le=1)
    
    способ_свободная: int = Field(0, ge=0, le=1)
    способ_альтернативная: int = Field(0, ge=0, le=1)
    способ_ипотека: int = Field(0, ge=0, le=1)
    способ_переуступка: int = Field(0, ge=0, le=1)
    способ_договор_долевого_участия: int = Field(0, ge=0, le=1)
    
    в_доме_газ: int = Field(0, ge=0, le=1)
    в_доме_мусоропровод: int = Field(0, ge=0, le=1)
    в_доме_консьерж: int = Field(0, ge=0, le=1)
    
    детская_площадка: int = Field(0, ge=0, le=1)
    спортивная_площадка: int = Field(0, ge=0, le=1)
    закрытая_территория: int = Field(0, ge=0, le=1)
    
    парковка_подземная: int = Field(0, ge=0, le=1)
    парковка_открытая_во_дворе: int = Field(0, ge=0, le=1)
    парковка_за_шлагбаумом: int = Field(0, ge=0, le=1)
    парковка_наземная_многоуровневая: int = Field(0, ge=0, le=1)

    @model_validator(mode='after')
    def validate_apartment_studio(self):
        """Валидация связей между апартаментами и студией"""
        # Если студия, то комната = 0.7
        if self.студия == 1:
            self.комнат = 0.7
        return self


def load_model_and_processor():
    """Загружаем модель и процессор"""
    try:
        # Загружаем CatBoost модель
        model = joblib.load("artifacts/catboost_model.joblib")
        
        # Загружаем артефакты инференса
        inference_artifacts = joblib.load("artifacts/inference_artifacts.joblib")
        
        # Загружаем артефакты препроцессинга
        artifacts = joblib.load("artifacts/preprocessing_artifacts.joblib")
        
        # Создаем процессор
        processor = SingleFlatProcessor()
        
        if processor is None:
            raise ValueError("Процессор не создан")
        
        return {
            "model": model,
            "processor": processor,
            "inference_artifacts": inference_artifacts,
            "artifacts": artifacts
        }
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке: {e}")
        raise


def predict_with_model(app_state: Dict[str, Any], processed_data: pd.DataFrame):
    """Делаем предсказание с использованием модели"""
    
    model = app_state['model']
    inference_artifacts = app_state['inference_artifacts']
    
    # Получаем имена признаков
    feature_columns = inference_artifacts['feature_columns']
    
    # Проверяем наличие необходимых признаков
    available_features = set(processed_data.columns)
    missing_features = set(feature_columns) - available_features
    
    if missing_features:
        # Добавляем отсутствующие признаки со значением 0
        for feature in missing_features:
            processed_data[feature] = 0
    
    # Выбираем только нужные признаки в правильном порядке
    processed_data = processed_data[feature_columns]
    
    # Преобразуем категориальные признаки в строки
    categorical_cols = inference_artifacts['categorical_cols']
    for col in categorical_cols:
        if col in processed_data.columns:
            processed_data[col] = processed_data[col].astype(str).fillna('неизвестно')
    
    # Предсказание модели (логарифмированная цена)
    prediction_log = float(model.predict(processed_data)[0])
    
    # Экспоненцируем обратно к исходной шкале
    prediction = np.expm1(prediction_log)
    
    # Ограничиваем максимальную цену
    prediction = np.clip(prediction, 0, MAX_PRICE)
    
    return float(prediction)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan контекст для управления жизненным циклом приложения"""
    logger.info("Запуск приложения...")
    
    try:
        models_dict = load_model_and_processor()
        app.state.models = models_dict
        logger.info("Модель успешно загружена")
    except Exception as e:
        logger.error(f"Ошибка загрузки модели: {e}")
        app.state.models = None
    
    yield
    
    logger.info("Завершение работы приложения...")


app = FastAPI(
    title="Калькулятор стоимости квартиры",
    version="1.0.0",
    lifespan=lifespan
)

templates = Jinja2Templates(directory=".")


def get_app_state(request: Request) -> Dict[str, Any]:
    return request.app.state.models


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/predict")
async def api_predict(flat_data: FlatInput, request: Request):
    try:
        app_state = get_app_state(request)
        if app_state is None:
            raise HTTPException(status_code=500, detail="Модель не загружена")
        
        # Преобразуем Pydantic модель в dict
        data = flat_data.dict()
        
        # Обрабатываем данные
        processor = app_state['processor']
        processed_data = processor.process_single_flat(data)
        
        # Делаем предсказание
        prediction = predict_with_model(app_state, processed_data)
        
        # Форматируем результат
        formatted_price = f"{prediction:,.0f}".replace(",", " ")
        price_millions = f"{prediction/1_000_000:.2f}"
        
        return {
            "status": "success",
            "predicted_price": float(prediction),
            "predicted_price_formatted": f"{formatted_price} руб.",
            "predicted_price_millions": price_millions
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/health")
async def health_check(request: Request):
    app_state = get_app_state(request)
    if app_state is not None:
        return {"status": "healthy", "model_loaded": True}
    else:
        return {"status": "unhealthy", "model_loaded": False}


@app.get("/predict")
async def redirect_to_home():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/")