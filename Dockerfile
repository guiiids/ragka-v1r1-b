FROM python:3.11-slim

WORKDIR /app

# Pre-copy only requirements to leverage layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt azure-storage-blob

# Copy the rest of the app
COPY . .

EXPOSE 8000

CMD ["gunicorn", "main:app", "--bind", "0.0.0.0:8000"]

RUN apt-get update && apt-get install -y openssh-server \
    && mkdir /var/run/sshd
    
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config

ENV PORT 2222

COPY startup.sh /startup.sh
RUN chmod +x /startup.sh

CMD ["/startup.sh", "gunicorn", "main:app", "--bind", "0.0.0.0:8000"]
