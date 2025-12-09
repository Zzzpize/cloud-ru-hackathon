FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl


COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml

COPY src/ .

EXPOSE 8000

CMD ["uv", "run", "python", "main.py"]