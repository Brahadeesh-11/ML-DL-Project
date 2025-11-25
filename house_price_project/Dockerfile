# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# ensure model exists, otherwise user should run train_and_save.py before building or copy model.pkl into the image
CMD ["gunicorn", "houseprice_project.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2"]
