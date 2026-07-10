# StockWise AI

> **An AI-powered Stock Recommendation & Portfolio Management Platform built using Machine Learning, Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and Real-Time Market Data.**

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-ML-orange?style=for-the-badge&logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-AI-yellow?style=for-the-badge&logo=scikitlearn)
![Groq](https://img.shields.io/badge/Groq-LLM-red?style=for-the-badge)
![SQLite](https://img.shields.io/badge/SQLite-Database-blue?style=for-the-badge&logo=sqlite)

</p>

---

# Live Demo

> **Coming Soon**

---

# Screenshots

## Home Dashboard

> _(Insert Screenshot Here)_

---

## AI Recommendations

> _(Insert Screenshot Here)_

---

## Portfolio Dashboard

> _(Insert Screenshot Here)_

---

## Forecast Graph

> _(Insert Screenshot Here)_

---

## AI Chat Assistant

> _(Insert Screenshot Here)_

---

## Market News

> _(Insert Screenshot Here)_

---

## SMS Recommendation

> _(Insert Screenshot Here)_

---

# Overview

StockWise AI is an intelligent stock recommendation platform designed for the Indian stock market.

Unlike traditional stock screeners that only display historical data, StockWise AI combines:

- Machine Learning
- Deep Learning
- Retrieval-Augmented Generation (RAG)
- Large Language Models (Groq)
- Technical Analysis
- Live Market News
- Portfolio Analytics

to provide intelligent, explainable, AI-powered investment recommendations.

The platform predicts future stock movements, generates buy recommendations, manages portfolios, forecasts prices, and answers financial questions using a context-aware AI assistant.

---

# Key Features

## AI Stock Recommendation Engine

- Daily Buy Recommendations
- Expected Return Prediction
- Buy Confidence Score
- Technical Indicator Analysis
- Multi-stock Ranking
- Automatic Daily Recommendation Generation

---

## AI Financial Assistant

Powered using:

- Groq LLM
- RAG
- Live Market News
- Portfolio Context

Example Questions:

```
Why is Reliance falling today?

Compare Infosys and TCS.

Should I buy SBI now?

How do western markets affect Indian stocks?

Explain today's market movement.
```

---

## Price Forecasting

Forecast periods:

- 1 Week
- 1 Month
- 3 Months
- 6 Months

Uses:

- TensorFlow
- Scikit-Learn
- Historical OHLCV Data
- Technical Indicators

---

## Portfolio Management

- Add Holdings
- Delete Holdings
- Live Profit & Loss
- Portfolio Performance
- Historical Recommendations

---

## Live Market Data

Supports

- Yahoo Finance
- TwelveData (Fallback)

---

## Market News

Retrieves

- Company News
- Sector News
- Market Headlines
- Economic Events

---

## Automated Scheduler

Automatically

- Updates Recommendations
- Sends SMS Alerts
- Refreshes Market Data
- Updates Database

---

# Walk-Forward Backtesting

The recommendation engine was evaluated using **walk-forward validation** to simulate real-world deployment without look-ahead bias.

## Current Results

| Metric                     | Result                       |
| -------------------------- | ---------------------------- |
| Directional Accuracy       | **62.5%**                    |
| Mean Absolute Error        | **1.01%**                    |
| Correlation                | **0.28**                     |
| Target Price Hit Rate      | **53.5%**                    |
| Average Daily Top-5 Return | **+0.20%**                   |
| Market Baseline            | **-0.058%**                  |
| Cumulative Return          | **+7.99% (40 Trading Days)** |

These results were generated across **40 real NSE trading sessions** using **200 prediction/outcome pairs**.

---

# Architecture

```
                                    USER

                                      │

                                      ▼

                            Flask Web Application

      ┌─────────────────────┬─────────────────────────┬─────────────────────┐

      ▼                     ▼                         ▼

 Recommendation Engine   Portfolio Manager      AI Chat Assistant

      │                     │                         │

      ▼                     ▼                         ▼

 TensorFlow Model      SQLite Database         Groq LLM

      │                                               │

      ▼                                               ▼

 Technical Indicators                        RAG Engine

      │                                               │

      ▼                                               ▼

 Historical Data                        Live Market News

      │

      ▼

 Daily Buy Recommendations
```

---

# System Workflow

```
User

 │

 ▼

Frontend (HTML/CSS/JS)

 │

 ▼

Flask Backend

 │

 ├──────────────► Portfolio Module

 │

 ├──────────────► Recommendation Engine

 │

 ├──────────────► Forecast Module

 │

 └──────────────► AI Chat Module

                         │

                         ▼

                Groq + RAG + News Tools

                         │

                         ▼

                 AI Generated Response
```

---

# Sequence Diagram

```
User

 │

 ▼

Open Dashboard

 │

 ▼

Flask

 │

 ▼

Load Portfolio

 │

 ▼

Recommendation Engine

 │

 ▼

ML Model Prediction

 │

 ▼

Rank Stocks

 │

 ▼

Display Recommendations

 │

 ▼

Ask AI Question

 │

 ▼

Groq LLM

 │

 ▼

Retrieve News

 │

 ▼

Generate Context

 │

 ▼

Return Response
```

---

# Model Training Pipeline

```
Historical Stock Data

        │

        ▼

Feature Engineering

        │

        ▼

Technical Indicators

        │

        ▼

Data Cleaning

        │

        ▼

Train/Test Split

        │

        ▼

Model Training

        │

        ▼

TensorFlow + Scikit Learn

        │

        ▼

Prediction

        │

        ▼

Ranking Engine

        │

        ▼

Daily Recommendations
```

---

# Backtesting Workflow

```
Historical Dataset

        │

        ▼

Train using Past Data

        │

        ▼

Predict Next Trading Day

        │

        ▼

Generate Top 5 Picks

        │

        ▼

Compare with Actual Returns

        │

        ▼

Calculate

Accuracy

MAE

Correlation

Hit Rate

Portfolio Returns

        │

        ▼

Repeat for 40 Trading Days
```

---

# Database Schema

```
+----------------------+

Users

-----------------------

id

username

email

password

+----------------------+

Portfolio

-----------------------

id

user_id

symbol

quantity

buy_price

purchase_date

+----------------------+

Recommendations

-----------------------

id

symbol

prediction

confidence

target_price

recommendation_date

+----------------------+

Chat History

-----------------------

id

user_id

query

response

timestamp
```

---

# REST API

## Portfolio

```
GET /portfolio
```

Returns portfolio.

---

```
POST /portfolio/add
```

Adds a stock.

---

```
DELETE /portfolio/delete
```

Deletes a stock.

---

## Forecast

```
GET /get_forecast?symbol=RELIANCE.NS
```

Returns forecast.

---

## Live Price

```
GET /get_current_stock_info
```

Returns

- Current Price
- Change %
- Volume
- High
- Low

---

## AI Chat

```
POST /api/ai/chat
```

Example Request

```json
{
  "message": "Should I buy Reliance today?"
}
```

---

## Recommendations

```
GET /recommendations
```

Returns latest recommendations.

---

# Tech Stack

## Backend

- Flask
- Python

## AI

- Groq LLM
- Retrieval Augmented Generation (RAG)

## Machine Learning

- TensorFlow
- Scikit-Learn

## Data

- Pandas
- NumPy

## Database

- SQLite

## APIs

- Yahoo Finance
- TwelveData

## Frontend

- HTML
- CSS
- JavaScript

---

# Project Structure

```
StockWiseAI/

│

├── app.py

├── recommender.py

├── ml_model.py

├── rag_engine.py

├── rag_chat_agent.py

├── market_tools.py

├── ai_routes.py

├── scheduler.py

├── db.py

├── msg.py

├── schema.sql

├── templates/

├── requirements.txt

├── Dockerfile

├── README.md

└── .env.example
```

---

# Installation

Clone

```bash
git clone https://github.com/AaryaButolia11/StockWiseAI.git

cd StockWiseAI
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

Install Packages

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env`

```env
GROQ_API_KEY=

TWILIO_ACCOUNT_SID=

TWILIO_AUTH_TOKEN=

TWILIO_PHONE=

SECRET_KEY=
```

---

# Run

```bash
python app.py
```

Open

```
http://127.0.0.1:8080
```

---

# Future Improvements

- PostgreSQL Support
- Docker Compose
- Kubernetes Deployment
- OAuth Login
- News Sentiment Analysis
- Reinforcement Learning Portfolio Optimization
- Explainable AI Dashboard
- Candlestick Pattern Detection
- Email Alerts
- Mobile Application
- Multi-user Authentication
- Risk Profiling
- Watchlist Feature

---

# Resume Highlights

- AI-powered stock recommendation engine using Machine Learning and Technical Indicators.
- Integrated Groq-powered RAG chatbot for financial question answering.
- Built end-to-end Flask application with portfolio management and live market analytics.
- Implemented walk-forward backtesting with **62.5% directional accuracy**, **53.5% target hit rate**, and **+7.99% cumulative return** over **40 NSE trading days**.
- Automated recommendation scheduling and SMS notifications.
- Integrated multiple market data providers with fallback handling.

---

# Author

**Aarya Butolia**

B.Tech CSE (AI & ML)

VIT Bhopal University

GitHub:
https://github.com/AaryaButolia11

LinkedIn:
https://www.linkedin.com/in/YOUR_LINKEDIN

---

# License

MIT License

---

# Star the Repository

If you found this project helpful, consider giving it a ⭐ on GitHub!
