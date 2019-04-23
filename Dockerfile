FROM python:3.6
ADD ./ /project

WORKDIR /project
ENV PYTHONPATH /project
RUN apt-get update && apt-get -y upgrade
RUN apt-get --assume-yes install libopenmpi-dev

#RUN pip install Click
RUN pip install bayesian-optimization
RUN pip install stable-baselines
RUN pip install tensorflow
RUN pip install torch
RUN pip install gym
RUN pip install pygame
RUN pip install pymunk
RUN pip install matplotlib
RUN pip install pymongo
RUN pip install scipy
#RUN pip install -r requirements.txt
CMD [ "python3", "./docker_run.py" ]
