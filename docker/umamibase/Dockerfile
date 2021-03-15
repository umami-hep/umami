FROM UPDATED_IN_GITLAB_CI_JOB

## ensure locale is set during build
ENV LANG C.UTF-8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y git debconf-utils h5utils && \
    echo "krb5-config krb5-config/add_servers_realm string CERN.CH" | debconf-set-selections && \
    echo "krb5-config krb5-config/default_realm string CERN.CH" | debconf-set-selections && \
    apt-get install -y krb5-user && \
    apt-get install -y vim emacs less screen graphviz python3-tk wget 

COPY requirements.txt .

RUN pip install -r requirements.txt