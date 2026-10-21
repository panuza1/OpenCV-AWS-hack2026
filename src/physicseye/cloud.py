"""Direct AWS Lambda invocation and deterministic local/cloud parity checks."""

import base64
import json
import math
import os
import re
import time
from datetime import datetime, timezone

import boto3
from botocore.config import Config

from physicseye.physics import predict_trajectory


MODELS = {
    "constant_velocity",
    "fitted_deceleration",
    "frozen_deceleration",
    "adaptive_deceleration",
}
_DURATION = re.compile(r"Duration:\s*([0-9.]+)\s*ms")
_BILLED_DURATION = re.compile(r"Billed Duration:\s*([0-9.]+)\s*ms")


class LambdaInvocationError(RuntimeError):
    """Raised when Lambda cannot supply a valid PhysicsEye response."""


def prediction_response_to_result(response):
    required = {
        "status", "predictions", "parameters", "adaptive_trace", "warnings", "engine"
    }
    if not isinstance(response, dict) or not required.issubset(response):
        raise LambdaInvocationError("Lambda response does not match the PhysicsEye schema")
    if response["status"] != "ok":
        raise LambdaInvocationError(response.get("error", "Lambda prediction failed"))
    if response["engine"] != "aws_lambda" or set(response["predictions"]) != MODELS:
        raise LambdaInvocationError("Lambda response has an invalid engine or model set")
    for name, model in response["predictions"].items():
        if response["parameters"].get(name) != model.get("parameters"):
            raise LambdaInvocationError("Lambda response parameters do not match predictions")
    return {
        "status": "ok",
        "cutoff_time_s": response["cutoff_time_s"],
        "fitting_window": response["fitting_window"],
        "models": response["predictions"],
        "adaptive_state": response["adaptive_trace"],
        "warnings": response["warnings"],
    }


def _log_durations(encoded_log):
    if not encoded_log:
        return None, None
    try:
        log = base64.b64decode(encoded_log).decode("utf-8", errors="replace")
    except (ValueError, TypeError):
        return None, None
    duration = _DURATION.search(log)
    billed = _BILLED_DURATION.search(log)
    return (
        float(duration.group(1)) if duration else None,
        float(billed.group(1)) if billed else None,
    )


def invoke_lambda(observations, cutoff_time_s, horizons_s, *, client=None,
                  function_name=None, region=None):
    function_name = function_name or os.getenv("PHYSICSEYE_LAMBDA_NAME")
    region = region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    if not function_name:
        raise LambdaInvocationError("PHYSICSEYE_LAMBDA_NAME is not configured")
    if not region:
        raise LambdaInvocationError("AWS_REGION is not configured")
    if client is None:
        client = boto3.client(
            "lambda",
            region_name=region,
            config=Config(connect_timeout=3, read_timeout=20, retries={"max_attempts": 1}),
        )

    event = {
        "observations": observations,
        "cutoff_time_s": cutoff_time_s,
        "horizons_s": list(horizons_s),
    }
    started = time.perf_counter()
    response = client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        LogType="Tail",
        Payload=json.dumps(event, allow_nan=False).encode("utf-8"),
    )
    invocation_latency_ms = (time.perf_counter() - started) * 1000.0
    status_code = response.get("StatusCode")
    if status_code != 200:
        raise LambdaInvocationError(f"Lambda returned HTTP status {status_code}")
    payload_stream = response.get("Payload")
    payload = payload_stream.read() if hasattr(payload_stream, "read") else payload_stream
    try:
        decoded = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise LambdaInvocationError(f"Lambda returned invalid JSON: {error}") from error
    if response.get("FunctionError"):
        message = decoded.get("errorMessage", response["FunctionError"]) if isinstance(decoded, dict) else response["FunctionError"]
        raise LambdaInvocationError(f"Lambda function error: {message}")
    prediction = prediction_response_to_result(decoded)
    duration_ms, billed_duration_ms = _log_durations(response.get("LogResult"))
    response_metadata = response.get("ResponseMetadata", {})
    return {
        "prediction": prediction,
        "cloud_response": decoded,
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "function_name": function_name,
            "region": region,
            "request_id": response_metadata.get("RequestId"),
            "executed_version": response.get("ExecutedVersion"),
            "invocation_latency_ms": invocation_latency_ms,
            "lambda_duration_ms": duration_ms,
            "lambda_billed_duration_ms": billed_duration_ms,
        },
    }


def compare_predictions(local, remote, tolerance=1e-9):
    numeric_differences = []
    differences = []
    schema_mismatches = []

    def visit(left, right, path="result"):
        if isinstance(left, bool) or isinstance(right, bool):
            if left != right:
                schema_mismatches.append(path)
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            difference = abs(float(left) - float(right))
            numeric_differences.append(difference)
            if not math.isfinite(difference) or difference > tolerance:
                differences.append((path, difference))
        elif isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                schema_mismatches.append(path)
                return
            for key in left:
                visit(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                schema_mismatches.append(path)
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{path}[{index}]")
        elif left != right:
            schema_mismatches.append(path)

    visit(local, remote)
    max_difference = max(numeric_differences, default=0.0)
    return {
        "passed": not differences and not schema_mismatches,
        "tolerance": tolerance,
        "max_absolute_difference": max_difference,
        "numeric_mismatches": [item[0] for item in differences],
        "schema_mismatches": schema_mismatches,
    }


def execute_prediction(mode, observations, cutoff_time_s, horizons_s, **lambda_options):
    """Execute locally or on Lambda; AWS errors never fall back to local output."""
    history = [point for point in observations if point["time_s"] <= cutoff_time_s]
    started = time.perf_counter()
    local = predict_trajectory(history, cutoff_time_s, horizons_s)
    local_latency_ms = (time.perf_counter() - started) * 1000.0
    if local["status"] != "ok":
        return {"status": "error", "error": local["error"], "success_count": 0,
                "failure_count": 1, "local_latency_ms": local_latency_ms}
    if mode.upper() == "LOCAL":
        return {"status": "ok", "prediction": local, "engine": "local",
                "success_count": 1, "failure_count": 0,
                "local_latency_ms": local_latency_ms, "parity": None, "metadata": {}}
    if mode.upper() != "AWS":
        return {"status": "error", "error": f"unknown execution mode: {mode}",
                "success_count": 0, "failure_count": 1,
                "local_latency_ms": local_latency_ms}
    try:
        cloud = invoke_lambda(history, cutoff_time_s, horizons_s, **lambda_options)
    except Exception as error:  # boto3 exposes several credential/network exception classes.
        return {"status": "error", "error": str(error), "engine": "aws_lambda",
                "success_count": 0, "failure_count": 1,
                "local_latency_ms": local_latency_ms}
    return {
        "status": "ok",
        "prediction": cloud["prediction"],
        "engine": "aws_lambda",
        "success_count": 1,
        "failure_count": 0,
        "local_latency_ms": local_latency_ms,
        "parity": compare_predictions(local, cloud["prediction"]),
        "metadata": cloud["metadata"],
    }
