FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip3 install --no-cahce-dir -r requirements.txt

CMD ["python", "train.py", "benchmark=hotpotqa", "run_name=docker_mistral_hotpot_0", "testing=false", "resume=false"]