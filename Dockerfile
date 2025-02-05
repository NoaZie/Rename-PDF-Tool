# Verwende ein leichtes Python-Image
FROM python:3.9-slim

# Setze das Arbeitsverzeichnis
WORKDIR /app

# Kopiere die Abhängigkeiten und installiere sie
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den restlichen Code ins Image
COPY . .

# Standardmäßiger Startbefehl
CMD ["python", "main.py"]
