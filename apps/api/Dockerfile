FROM python:3.11-slim
WORKDIR /app/apps/api
COPY apps/api/requirements.txt /app/apps/api/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV CLS_HOST=0.0.0.0
CMD uvicorn app.main:app --host ${CLS_HOST} --port 8000
