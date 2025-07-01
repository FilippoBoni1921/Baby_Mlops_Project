FROM python:3.11-slim

# Copy all project files into /app in the container
COPY . /app

# Set working directory to /app (project root)
WORKDIR /app

# Install only the dependencies needed for inference
RUN pip install -r requirements_inference.txt

# Set locale to avoid Unicode issues (optional but good)
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Expose FastAPI default port
EXPOSE 8000

# Start the FastAPI app from the 'server/main.py' entry point
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
