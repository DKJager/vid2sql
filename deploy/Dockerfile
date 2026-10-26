# vid2sql container image for both Lambda functions (arm64, Python 3.12).
# Build is done by `sam build`; nothing here runs in this repository's tests.
FROM public.ecr.aws/lambda/python:3.12
COPY deploy/requirements-lambda.txt ${LAMBDA_TASK_ROOT}/requirements-lambda.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt
COPY v2s ${LAMBDA_TASK_ROOT}/v2s
COPY web ${LAMBDA_TASK_ROOT}/web
COPY deploy/__init__.py deploy/lambda_pipeline.py deploy/lambda_query.py ${LAMBDA_TASK_ROOT}/deploy/
CMD ["deploy.lambda_query.handler"]
