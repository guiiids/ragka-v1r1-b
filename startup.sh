#!/bin/sh
# Generate a fresh SAS token and write to /tmp/sas_token.env
python /app/generate_sas_token.py

# Source the token into the environment if available
if [ -f /tmp/sas_token.env ]; then
  . /tmp/sas_token.env
fi

# Execute passed command (e.g., gunicorn)
exec "$@"