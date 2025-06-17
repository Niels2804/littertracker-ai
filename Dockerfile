# Python Image
FROM python:3.12-slim

# Working DIR
WORKDIR /app

# Installing packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY the code
COPY . .

# Exposing the port (FastAPI uses port 8000)
EXPOSE 8000

# Start de server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
