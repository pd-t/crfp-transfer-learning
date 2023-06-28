ARG USER_ID=1000
ARG GROUP_ID=100

FROM ludwigai/ludwig-ray-gpu:0.7.4
RUN pip install poetry

ARG USER_ID
USER root
RUN usermod -u $USER_ID -m ray
USER ray