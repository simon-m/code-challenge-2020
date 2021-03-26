FROM drtools/dask:latest

USER root

COPY ./requirements.txt /opt/preprocess_split_dataset/requirements.txt

RUN pip install -r /opt/preprocess_split_dataset/requirements.txt

COPY . /opt/preprocess_split_dataset/

WORKDIR /opt/preprocess_split_dataset/
