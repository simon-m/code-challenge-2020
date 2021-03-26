FROM drtools/dask:latest

USER root

COPY ./requirements.txt /opt/generate_features/requirements.txt

RUN pip install -r /opt/generate_features/requirements.txt

COPY . /opt/generate_features/

WORKDIR /opt/generate_features/
