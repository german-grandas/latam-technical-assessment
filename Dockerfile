# syntax=docker/dockerfile:1.2
FROM python:latest
# put you docker configuration here
WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r requirements.txt


COPY . /app/
EXPOSE 8000

CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8000"]