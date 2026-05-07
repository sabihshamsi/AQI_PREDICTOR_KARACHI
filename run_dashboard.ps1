# Start FastAPI backend
Write-Host "Starting FastAPI backend on http://localhost:8000..." -ForegroundColor Green
Start-Process python -ArgumentList "app/api.py" -NoNewWindow

# Wait a few seconds for the API to start
Start-Sleep -Seconds 3

# Start Streamlit dashboard
Write-Host "Starting Streamlit dashboard on http://localhost:8501..." -ForegroundColor Green
streamlit run app/dashboard.py
