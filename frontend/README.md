# CycleCast Frontend

A responsive React + TypeScript dashboard for the CycleCast Prophet forecasting API.

## Requirements

- Node.js 20 or newer recommended
- CycleCast FastAPI backend running at `http://127.0.0.1:8000`

## Run locally on Windows PowerShell

```powershell
cd D:\cyclecast-mlops

# Copy this frontend folder into D:\cyclecast-mlops\frontend first.
cd frontend

Copy-Item .env.example .env
npm install
npm run dev
```

Open http://127.0.0.1:5173.

In another terminal, run the backend:

```powershell
cd D:\cyclecast-mlops
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## API configuration

By default, the UI calls `http://127.0.0.1:8000`. To use another local API URL, edit `.env`:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8000
```

Restart the Vite development server after changing environment variables.

The API must permit browser requests from `http://127.0.0.1:5173`. If your FastAPI app does not already enable CORS, add the following to `app/main.py` after creating `app = FastAPI(...)`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)
```

Place the import near the other imports and the middleware block immediately after the `FastAPI(...)` instance is created.

## Production build

```powershell
npm run build
npm run preview
```

The dashboard labels the forecasts as historical because the current saved Prophet models were trained on May 2024 data. It does not imply real-time bike availability.
