FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
WORKDIR /app
COPY . .
RUN pip install -r serve/requirements.txt
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
