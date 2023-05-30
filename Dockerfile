FROM ludwigai/ludwig-ray-gpu:0.7.4
RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN chmod +x create_minio_credetnials.sh cml-dvc-repro.sh
ENTRYPOINT ["create_minio_credentials.sh && ./cml-dvc-repro.sh"]

