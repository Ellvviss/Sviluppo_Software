FROM python:3.10-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y && rm -rf /var/lib/apt/lists/*

# Imposta la cartella di lavoro
WORKDIR /app

# Copia il codice nella cartella di lavoro
COPY . /app

# Copia delle dipendenze del progetto nel container
COPY requirements.txt .
# Installa le dipendenze
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia di tutto il progetto
COPY . .

# Esponi la porta per Gradio
EXPOSE 7860


ENV PYTHONPATH=/app
CMD ["python", "-u", "src/dogs_vs_cats_project.py"]