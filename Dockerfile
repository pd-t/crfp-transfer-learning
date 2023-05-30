FROM ludwigai/ludwig-ray-gpu:0.7.4

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN chmod +x create_minio_credetnials.sh cml-dvc-repro.sh
ENTRYPOINT ["create_minio_credentials.sh && ./cml-dvc-repro.sh"]

