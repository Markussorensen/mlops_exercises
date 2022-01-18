# Base image
FROM python:3.9-slim
# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*
# Copying over the essential files
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
# Setting workdir and installing dependencies
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
# Entrypoint, this is what is run when starting the application.
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
