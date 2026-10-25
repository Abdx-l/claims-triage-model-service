FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && useradd -u 10001 app
COPY . .
USER app
CMD ["flask","--app","app","run","--host=0.0.0.0"]
