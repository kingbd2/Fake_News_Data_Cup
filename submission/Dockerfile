FROM python:3

COPY sample_submission.py /usr/src/

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.7 \
    python3-pip \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install matplotlib 

CMD [ "python3", "/usr/src/sample_submission.py" ]