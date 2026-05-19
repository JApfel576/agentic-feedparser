# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.14.3
FROM python:${PYTHON_VERSION}-slim AS base

# copy uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set location for uv to leverage .venv without activation 
ENV UV_PROJECT_ENVIRONMENT="/usr/local" 
WORKDIR /app

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/go/dockerfile-user-best-practices/
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
uv sync --frozen --no-install-workspace --no-dev

COPY . /app

# locked after freeze to update uv.lock
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# create directory for volume data and give permissions to create folders and files
RUN mkdir -p /var/data \
&& chown -R appuser:appuser /var/data

# Switch to the non-privileged user to run the application.
USER appuser

# Copy the source code into the container.
COPY services/fastapi_app/src/fastapi_app/main.py /app

# Expose the port that the application listens on.
EXPOSE 8000 

# Run the application.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]