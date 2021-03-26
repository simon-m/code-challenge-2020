FROM drtools/dask:latest

USER root

COPY ./requirements.txt /opt/train_model/requirements.txt

RUN pip install -r /opt/train_model/requirements.txt

COPY . /opt/train_model/

WORKDIR /opt/train_model/
