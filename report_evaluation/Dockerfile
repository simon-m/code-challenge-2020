FROM drtools/dask:latest

USER root

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY ./requirements.txt /opt/report_evaluation/requirements.txt

RUN pip install -r /opt/report_evaluation/requirements.txt --ignore-installed

COPY . /opt/report_evaluation/

WORKDIR /opt/report_evaluation/
