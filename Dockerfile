FROM python:3.11-slim

WORKDIR /app

# Pre-copy only requirements to leverage layer caching
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

EXPOSE 8000
EXPOSE 2222


RUN apt-get update && apt-get install -y openssh-server \
    && mkdir /var/run/sshd
    
RUN echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
RUN echo 'Port 2222' >> /etc/ssh/sshd_config
RUN echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
RUN echo 'root:root' | chpasswd


COPY startup.sh /startup.sh
RUN sed -i 's/\r$//' /startup.sh && chmod +x /startup.sh

CMD ["/startup.sh", "gunicorn", "main:app", "--bind", "0.0.0.0:8000"]
