# PhysicsEye AWS Lambda deployment

PhysicsEye keeps video decoding and OpenCV 5 tracking on the local machine. The local application sends only timestamped observations through the selected cutoff to one private AWS Lambda function. Lambda imports the same pure Python `physicseye.physics` module used locally, reconstructs the adaptive state, and returns all four forecasts. No API Gateway, S3 bucket, database, layer, OpenCV wheel, or public endpoint is required.

The instructions use the supported `python3.13` managed runtime on Amazon Linux 2023. AWS documents the current [Python runtimes](https://docs.aws.amazon.com/lambda/latest/dg/lambda-python.html), [ZIP package layout](https://docs.aws.amazon.com/lambda/latest/dg/python-package.html), [execution roles](https://docs.aws.amazon.com/lambda/latest/dg/lambda-intro-execution-role.html), and [synchronous invocation](https://docs.aws.amazon.com/cli/latest/reference/lambda/invoke.html).

## Request and response

Direct invocation request:

```json
{
  "observations": [{"time_s": 0.0, "x_px": 40.0, "y_px": 120.0, "valid": true}],
  "cutoff_time_s": 2.0,
  "horizons_s": [0.25, 0.5, 1.0]
}
```

The handler accepts only those three top-level fields, at most 5,000 observations, at most 20 horizons, and at most 1,000,000 serialized bytes. The shared physics validator rejects unordered timestamps, non-finite values, malformed observations, and every observation after the cutoff.

A successful response contains `predictions`, per-model `parameters`, `adaptive_trace`, `warnings`, `engine=aws_lambda`, the cutoff/fitting window, schema version, and request context. Errors use `status=error` with empty prediction fields.

## 1. Configure the standard AWS credential chain

Install AWS CLI v2 and configure a profile, IAM Identity Center session, or role outside this repository. Do not put access keys in `.env` or source files.

```bash
export AWS_REGION=ap-southeast-1
export PHYSICSEYE_LAMBDA_NAME=physicseye-predict
export PHYSICSEYE_LAMBDA_ROLE_NAME=physicseye-lambda-role
aws sts get-caller-identity
```

The deployment identity needs the IAM actions used below and Lambda create/update/read actions. The dashboard identity needs only `lambda:InvokeFunction` on the deployed function ARN.

## 2. Create the execution role once

The function calls no other AWS service. Its execution role trusts Lambda and receives only the AWS managed basic logging policy, which AWS documents as the minimum CloudWatch Logs permission for standard Lambda logging.

```bash
aws iam create-role \
  --role-name "$PHYSICSEYE_LAMBDA_ROLE_NAME" \
  --assume-role-policy-document file://cloud/lambda-trust-policy.json

aws iam attach-role-policy \
  --role-name "$PHYSICSEYE_LAMBDA_ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

export PHYSICSEYE_LAMBDA_ROLE_ARN="$(aws iam get-role \
  --role-name "$PHYSICSEYE_LAMBDA_ROLE_NAME" \
  --query 'Role.Arn' --output text)"
```

The caller policy for the dashboard should be restricted to this function:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "lambda:InvokeFunction",
    "Resource": "arn:aws:lambda:REGION:ACCOUNT_ID:function:physicseye-predict"
  }]
}
```

Attach that policy to the user or role whose standard credentials run Streamlit.

## 3. Package and deploy

Packaging is offline and creates an approximately 8 KB dependency-free archive:

```bash
PYTHON=.venv/bin/python scripts/deploy_lambda.sh package
unzip -l dist/physicseye-lambda.zip
```

Deployment creates the function if absent and updates only its code otherwise:

```bash
scripts/deploy_lambda.sh deploy
aws lambda wait function-active-v2 \
  --region "$AWS_REGION" --function-name "$PHYSICSEYE_LAMBDA_NAME"
```

The created function uses Python 3.13, 128 MB memory, a 10-second timeout, x86_64, and handler `lambda_handler.handler`.

## 4. Invoke and verify

Create a causal request from an existing tracker export:

```bash
env -u PYTHONPATH .venv/bin/python - <<'PY'
import json

with open("outputs/slowing_down.track.json", encoding="utf-8") as stream:
    observations = json.load(stream)
cutoff = 2.0
request = {
    "observations": [point for point in observations if point["time_s"] <= cutoff],
    "cutoff_time_s": cutoff,
    "horizons_s": [0.25, 0.5, 1.0],
}
with open("/tmp/physicseye-request.json", "w", encoding="utf-8") as stream:
    json.dump(request, stream, allow_nan=False)
PY

aws lambda invoke \
  --region "$AWS_REGION" \
  --function-name "$PHYSICSEYE_LAMBDA_NAME" \
  --invocation-type RequestResponse --log-type Tail \
  --cli-binary-format raw-in-base64-out \
  --payload file:///tmp/physicseye-request.json \
  /tmp/physicseye-response.json

cat /tmp/physicseye-response.json
```

Run the dashboard with the same standard credential chain:

```bash
export AWS_REGION=ap-southeast-1
export PHYSICSEYE_LAMBDA_NAME=physicseye-predict
env -u PYTHONPATH .venv/bin/streamlit run app.py
```

Choose **AWS**, analyze a video, then press **Invoke AWS Lambda**. The page reports request ID, region, function name, invocation latency, Lambda duration when returned by `LogType=Tail`, and deterministic parity against a local reference. Because both sides execute the same pure Python source and JSON preserves Python floating-point values, the parity tolerance is `1e-9`; schema keys must match exactly. Invocation metadata is included in the downloadable analysis JSON. An AWS error is displayed and never replaced with a local result.

## 5. Capture evidence

Record the displayed UTC timestamp, region, function identifier, AWS request ID, invocation and Lambda duration, parity result/max difference, adaptive update count, and predicted values. Preserve the Lambda response and dashboard screenshot. Mocked test metadata is not cloud evidence.

## 6. Cleanup

These commands delete the function and its dedicated execution role:

```bash
aws lambda delete-function \
  --region "$AWS_REGION" --function-name "$PHYSICSEYE_LAMBDA_NAME"

aws iam detach-role-policy \
  --role-name "$PHYSICSEYE_LAMBDA_ROLE_NAME" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam delete-role --role-name "$PHYSICSEYE_LAMBDA_ROLE_NAME"
```

Remove the restricted caller policy separately from whichever identity received it. CloudWatch log groups can remain after function deletion; inspect and delete `/aws/lambda/physicseye-predict` if retention is no longer wanted.
