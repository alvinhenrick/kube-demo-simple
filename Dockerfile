FROM python:3.6

RUN pip install --upgrade tensorflow==1.10.1
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN mkdir /code

WORKDIR /code

COPY / /code

CMD python train_and_evaluate.py