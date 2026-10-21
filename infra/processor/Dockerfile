# Build context = repo root (see .dockerignore). sam build: Metadata in infra/template.yaml.
FROM public.ecr.aws/lambda/python:3.12

COPY infra/processor/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Shared CPCB-style AQI package, then the air module (its only dependency is cpcb-aqi).
COPY shared/ /tmp/shared/
RUN pip install --no-cache-dir /tmp/shared && rm -rf /tmp/shared
COPY backend/src/air/ ${LAMBDA_TASK_ROOT}/air/

COPY infra/processor/*.py ${LAMBDA_TASK_ROOT}/
# smokefire.onnx, head.npz, export_report.json (gitignored; copied from the ML run).
COPY infra/processor/models/ ${LAMBDA_TASK_ROOT}/models/

CMD ["handler.handler"]
