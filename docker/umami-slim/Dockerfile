# This image is only meant for plotting etc., i.e. for everything without tensorflow
# dependence
FROM ${BASE_IMAGE}

## ensure locale is set during build
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y git debconf-utils h5utils && \
    apt-get install -y vim less graphviz python3-tk wget


COPY . /umami
WORKDIR /umami

RUN pip install -r requirements.txt
RUN cd /umami && python -m pip install .

