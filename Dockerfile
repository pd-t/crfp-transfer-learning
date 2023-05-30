FROM ludwigai/ludwig-ray-gpu:0.7.4 AS base
RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/* \
&& pip install poetry==1.4.2

FROM base AS python-environment
COPY *.toml *.lock ./
RUN poetry config virtualenvs.create false && poetry install
WORKDIR /app

FROM python-environment AS cml-dvc-repro
COPY . .
RUN chmod +x cml-dvc-repro.sh
ENTRYPOINT ["/app/cml-dvc-repro.sh"]

