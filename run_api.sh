#!/bin/bash
# Run FastAPI backend
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
