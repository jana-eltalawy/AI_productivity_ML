# AI Adoption → Productivity Predictor

This turns your notebook (`User_AI_prediction.ipynb`) into a real app with
three pieces:

| File               | Purpose                                                            |
|---------------------|---------------------------------------------------------------------|
| `train_model.py`    | Reproduces your notebook's preprocessing + trains a Random Forest  |
| `main.py`            | FastAPI backend that loads the trained model and serves `/predict` |
| `app.py`              | Streamlit UI that collects inputs and calls the FastAPI backend    |

The model used is `RandomForestRegressor` (it was one of the strongest
performers in your notebook's comparison and is simple to serve). Final
features it's trained on:

- **Numeric:** `Daily_Token_Usage`, `Tasks_Automated_Per_Week`, `Tokens_per_Task` (engineered)
- **Categorical (one-hot):** `Industry`, `Job_Role`, `Location`, `Primary_AI_Tool`, `Experience_Level` (engineered from `Experience_Years`)
- **Target:** `Productivity_Gain_Percent`

`Experience_Years` and `Adoption_Date` are only used to *derive* other
features and aren't fed to the model directly — same as in your notebook.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Train the model

Place your dataset CSV (e.g. `user_level_ai_adoption.csv`) next to
`train_model.py`, then run:

```bash
python train_model.py --csv user_level_ai_adoption.csv --out artifacts
```

This creates an `artifacts/` folder with:

- `model.joblib`
- `scaler.joblib`
- `feature_columns.joblib`

Re-run this script any time you want to retrain (new data, tweaked
hyperparameters, etc.) — it will overwrite the files in `artifacts/`.

## 3. Start the FastAPI backend

```bash
uvicorn main:app --reload --port 8000
```

- Health check: `GET http://127.0.0.1:8000/health`
- Interactive docs: `http://127.0.0.1:8000/docs`
- Prediction endpoint: `POST http://127.0.0.1:8000/predict`

Example request body:

```json
{
  "Industry": "Technology",
  "Job_Role": "Software Engineer",
  "Location": "Remote",
  "Primary_AI_Tool": "ChatGPT",
  "Experience_Years": 4,
  "Daily_Token_Usage": 15000,
  "Tasks_Automated_Per_Week": 12
}
```

`main.py` looks for `artifacts/` in the same directory by default. If you
keep artifacts elsewhere, set the `ARTIFACT_DIR` environment variable
before starting uvicorn:

```bash
ARTIFACT_DIR=/path/to/artifacts uvicorn main:app --reload --port 8000
```

## 4. Start the Streamlit frontend

In a **second terminal** (keep the FastAPI server running):

```bash
streamlit run app.py
```

This opens a form in your browser. Streamlit sends the form data as JSON
to `http://127.0.0.1:8000/predict` (via the `requests` library) and shows
the predicted `Productivity_Gain_Percent`.

If your FastAPI service runs somewhere other than `127.0.0.1:8000`
(e.g. deployed to a server), set the `FASTAPI_URL` environment variable
before launching Streamlit:

```bash
FASTAPI_URL=https://your-api.example.com streamlit run app.py
```

## How the three pieces connect

```
 CSV data
    │
    ▼
train_model.py  ──▶  artifacts/ (model.joblib, scaler.joblib, feature_columns.joblib)
                                │
                                ▼
                        main.py (FastAPI, port 8000)
                                ▲
                                │  HTTP POST /predict (JSON)
                                │
                        app.py (Streamlit, port 8501)
                                │
                                ▼
                             Browser (user fills form)
```

- **`train_model.py` → artifacts**: run once (or whenever you retrain);
  produces the files the API depends on.
- **`main.py` → artifacts**: loads them at startup and exposes `/predict`.
- **`app.py` → `main.py`**: makes an HTTP call from the Streamlit process
  to the FastAPI process — they can run on the same machine or different
  machines, as long as `FASTAPI_URL` points to the right place.

## Notes / things you may want to adjust

- The API currently accepts free-text strings for `Industry`, `Job_Role`,
  `Location`, and `Primary_AI_Tool`. Any value not seen during training is
  handled gracefully (it just won't trigger a category-specific effect,
  since one-hot columns are reindexed and missing ones default to 0). If
  you'd rather restrict inputs to the exact categories seen in training,
  swap the free-text fields in `app.py` for `st.selectbox` populated from
  your dataset's unique values, and consider adding an `Enum` in
  `main.py`'s `PredictionRequest` for stricter validation.
- CORS isn't needed because Streamlit calls FastAPI server-to-server (not
  from the browser's JavaScript). If you build a separate JS frontend
  instead, you'll need to add `fastapi.middleware.cors.CORSMiddleware`.
