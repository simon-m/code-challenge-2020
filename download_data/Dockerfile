FROM python:3.6-alpine
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY ./requirements.txt /opt/download_data/requirements.txt

RUN pip install -r /opt/download_data/requirements.txt

COPY ./ /opt/download_data/

WORKDIR /opt/download_data/
