FROM python:3.12

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    cython3 \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e '.[dev,notebook]'
