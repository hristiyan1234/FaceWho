FROM python:3.10-slim

# Инсталиране на минимални системни зависимости, включително компилатори и cmake
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Инсталиране на numpy предварително, за да може dlib да го намери
RUN pip install --no-cache-dir numpy==1.26.4

# Инсталиране на dlib без кеширане, за да пести памет
RUN pip install --no-cache-dir dlib==19.24.2

# Инсталиране на останалите изисквания
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
