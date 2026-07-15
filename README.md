.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000


Set-Location 'd:\projects\python_poject\earnings-call-backend'; .\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000


#to clear the database 
curl -X POST http://localhost:8000/rag/clear