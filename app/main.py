# app/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.model_service import load_models
from app.routers import forecast

app = FastAPI(
    title="Air Quality Forecasting API",
    description="Time-series forecasting for air pollution (COMP9417 project extension).",
    version="0.1.0",
)

# ---- CORS 配置（开发环境直接全开放最省事）----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # 开发阶段直接用 *，部署时可以改成特定域名
    allow_credentials=True,
    allow_methods=["*"],          # 允许所有方法：GET, POST, OPTIONS, ...
    allow_headers=["*"],          # 允许所有头
)


@app.on_event("startup")
def startup_event():
    # 启动时加载训练好的模型
    load_models("models")


@app.get("/health")
def health_check():
    return {"status": "ok"}


# 挂载预测路由
app.include_router(forecast.router, prefix="/api")
