!/usr/bin/env bash
export FLASK_APP=run.py
export FLASK_DEBUG=1
pip install -r requirements.txt
gunicorn run:flask_app/app -b 127.0.0.1:8000 -w 4 --access-logfile ./flask_app/logs/logs/access.log --error-logfile ./flask_app/logs/error.log