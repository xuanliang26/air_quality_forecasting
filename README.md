# Air Quality Forecasting API & Agent  
# 空气质量预测 API 与智能助手

## 1. Project Overview | 项目概览

This repository extends our COMP9417 course project into a more practical **backend + AI agent** application.

We use the **UCI Air Quality dataset** to train time-series models that forecast pollutant levels (CO, NMHC, C6H6, NOx, NO2).  
On top of the models, we build a **FastAPI-based backend** and a simple **LLM-powered “air quality assistant”**, so that predictions can be accessed via REST APIs or natural-language queries.

本仓库是在 COMP9417 课程项目基础上的扩展版本，重点放在 **后端服务化 + AI 智能助手** 上。

我们使用 **UCI 空气质量数据集** 训练时序模型，预测多种污染物的未来浓度（CO、NMHC、C6H6、NOx、NO2）。  
在此基础上，构建了一个基于 **FastAPI 的后端服务**，并集成了一个简单的 **LLM 智能助手**，支持通过 REST API 或自然语言提问获取预测结果和解释。

---

## 2. Main Features | 主要功能

- **Time-series regression models** for multi-horizon forecasting  
  多步污染物浓度预测的时序回归模型（1 / 6 / 12 / 24 小时）

- **Modular pipeline** for data preprocessing, feature engineering, training and evaluation  
  模块化的数据预处理、特征工程、训练与评估流程

- **RESTful API (FastAPI)** to expose prediction and metrics endpoints  
  使用 FastAPI 暴露预测与评估接口，提供 RESTful API 服务

- **AI assistant endpoint** that combines model outputs and an LLM to answer questions in natural language  
  将模型输出与大模型结合的智能问答接口，可以用自然语言询问未来空气质量和模型表现

---

## 3. Project Structure | 项目结构

```text
air-quality-forecasting/
|
├─AirQualityUCI.csv 
├─clean_air_quality.csv
├─ models/
│  ├─ regressor_co.pkl
│  ├─ regressor_nox.pkl
│  └─ scaler_features.pkl
│
├─ notebooks/
│  └─ regression_models.ipynb
│
├─ app/
│  ├─ config.py
│  ├─ schema.py
│  ├─ features.py
│  ├─ model_service.py
│  ├─ agent_service.py
│  ├─ routers/
│  │  ├─ forecast.py
│  │  ├─ metrics.py
│  │  └─ agent.py
│  └─ main.py
│
├─ data_process.py
├─ train_regression.py
│
├─ tests/
│  ├─ test_features.py
│  ├─ test_model_service.py
│  └─ test_api.py
│
├─ .env.example
└─ README.md

Typical requirements include:
fastapi
uvicorn
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
python-dotenv
requests
