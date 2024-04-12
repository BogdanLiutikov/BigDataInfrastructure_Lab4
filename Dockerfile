FROM python:3.10-slim

ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

RUN pip freeze
CMD ["python", "src/server.py", "-m", "RandomForestClassifier"]