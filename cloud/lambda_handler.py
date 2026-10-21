"""Dependency-free AWS Lambda adapter for the shared PhysicsEye engine."""

import json

from physicseye.physics import predict_trajectory


MAX_REQUEST_BYTES = 1_000_000
MAX_OBSERVATIONS = 5_000
MAX_HORIZONS = 20
REQUEST_FIELDS = {"observations", "cutoff_time_s", "horizons_s"}


def _error(message):
    return {
        "status": "error",
        "error": message,
        "predictions": {},
        "parameters": {},
        "adaptive_trace": None,
        "warnings": [],
        "engine": "aws_lambda",
    }


def _validate_event(event):
    if not isinstance(event, dict):
        raise ValueError("event must be a JSON object")
    unknown = set(event) - REQUEST_FIELDS
    missing = REQUEST_FIELDS - set(event)
    if missing:
        raise ValueError(f"missing request fields: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"unexpected request fields: {', '.join(sorted(unknown))}")
    try:
        size = len(json.dumps(event, allow_nan=False).encode("utf-8"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"event must contain only finite JSON values: {error}") from error
    if size > MAX_REQUEST_BYTES:
        raise ValueError(f"request exceeds {MAX_REQUEST_BYTES} bytes")
    if not isinstance(event["observations"], list):
        raise ValueError("observations must be a JSON array")
    if len(event["observations"]) > MAX_OBSERVATIONS:
        raise ValueError(f"observations exceeds the limit of {MAX_OBSERVATIONS}")
    if not isinstance(event["horizons_s"], list):
        raise ValueError("horizons_s must be a JSON array")
    if len(event["horizons_s"]) > MAX_HORIZONS:
        raise ValueError(f"horizons_s exceeds the limit of {MAX_HORIZONS}")


def handler(event, context):
    """Validate a direct-invocation event and run the shared causal engine."""
    try:
        _validate_event(event)
    except ValueError as error:
        return _error(str(error))

    result = predict_trajectory(
        event["observations"], event["cutoff_time_s"], event["horizons_s"]
    )
    if result["status"] != "ok":
        return _error(result["error"])

    return {
        "status": "ok",
        "cutoff_time_s": result["cutoff_time_s"],
        "fitting_window": result["fitting_window"],
        "predictions": result["models"],
        "parameters": {
            name: model["parameters"] for name, model in result["models"].items()
        },
        "adaptive_trace": result["adaptive_state"],
        "warnings": result["warnings"],
        "engine": "aws_lambda",
        "schema_version": 1,
        "request": {
            "id": getattr(context, "aws_request_id", None),
            "function_version": getattr(context, "function_version", None),
        },
    }
