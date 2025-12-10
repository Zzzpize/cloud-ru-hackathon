FROM python:3.12-slim

# Устанавливаем curl (для uv) и ffmpeg (для аудио)
RUN apt-get update && apt-get install -y curl ffmpeg && rm -rf /var/lib/apt/lists/*

# Устанавливаем uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Сначала зависимости (для кэширования слоев)
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml

# Потом код
COPY src/ .

EXPOSE 8000

CMD ["uv", "run", "python", "main.py"]