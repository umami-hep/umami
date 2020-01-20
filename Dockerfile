FROM tensorflow/tensorflow:latest-gpu-py3

## ensure locale is set during build
ENV LANG C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install numpy && \
    pip install pandas && \
    pip install dask && \
    pip install keras && \
    pip install uproot && \
    pip install jupyter && \
    pip install jupyterlab && \
    pip install matplotlib && \
    pip install seaborn && \
    pip install hep_ml && \
    pip install sklearn && \
    pip install tables && \
    pip install papermill pydot Pillow


RUN apt-get update && \
    apt-get install -y git debconf-utils h5utils && \
    echo "krb5-config krb5-config/add_servers_realm string CERN.CH" | debconf-set-selections && \
    echo "krb5-config krb5-config/default_realm string CERN.CH" | debconf-set-selections && \
    apt-get install -y krb5-user && \
    apt-get install -y vim less screen graphviz python3-tk wget

COPY . /umami
WORKDIR /umami

RUN python /umami/setup.py install 
