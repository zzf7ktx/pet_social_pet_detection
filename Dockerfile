FROM python:3.8.1
FROM pytorch/pytorch

ENV APP_HOME /app
WORKDIR $APP_HOME

COPY . /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "app.py"]