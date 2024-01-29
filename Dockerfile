FROM python:3.8-alpine

COPY ./requirements.txt /requirements.txt

RUN apk add --update \
    bash \
    nodejs \
    npm 

RUN python -m venv what_if_chatbot

RUN source what_if_chatbot/bin/activate

RUN pip install --upgrade pip
RUN pip install -r requirements.txt 

COPY ./RasaUI/package.json /RasaUI/package.json

RUN npm install react-scripts

COPY . .