import base64
import io
import json
import subprocess
import zipfile

import pytest

from cloud.lambda_handler import handler
from physicseye.cloud import execute_prediction
from physicseye.physics import predict_trajectory


def _observations():
    times = [0.0, 0.1, 0.22, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1]
    observations = []
    for index, time_s in enumerate(times):
        if index == 3:
            observations.append({"time_s": time_s, "x_px": None, "y_px": None, "valid": False})
        else:
            observations.append({
                "time_s": time_s,
                "x_px": 10.0 + 40.0 * time_s - 2.0 * time_s**2,
                "y_px": 20.0,
                "valid": True,
            })
    return observations


def _event(observations=None):
    return {
        "observations": observations or _observations(),
        "cutoff_time_s": 1.1,
        "horizons_s": [0.25, 0.5],
    }


class Context:
    aws_request_id = "local-test-request"
    function_version = "$LATEST"


class FakeLambda:
    def __init__(self, function_error=None):
        self.function_error = function_error
        self.calls = []

    def invoke(self, **kwargs):
        self.calls.append(kwargs)
        event = json.loads(kwargs["Payload"])
        response = handler(event, Context())
        result = {
            "StatusCode": 200,
            "Payload": io.BytesIO(json.dumps(response).encode()),
            "LogResult": base64.b64encode(
                b"REPORT RequestId: test Duration: 1.23 ms Billed Duration: 2 ms"
            ).decode(),
            "ExecutedVersion": "$LATEST",
            "ResponseMetadata": {"RequestId": "aws-test-request"},
        }
        if self.function_error:
            result["FunctionError"] = self.function_error
            result["Payload"] = io.BytesIO(b'{"errorMessage":"boom"}')
        return result


def test_valid_handler_matches_local_schema_and_values():
    event = _event()
    response = handler(event, Context())
    local = predict_trajectory(event["observations"], 1.1, [0.25, 0.5])

    assert response["status"] == "ok"
    assert response["engine"] == "aws_lambda"
    assert response["predictions"] == local["models"]
    assert response["adaptive_trace"] == local["adaptive_state"]
    assert response["request"]["id"] == "local-test-request"
    assert set(response["parameters"]) == set(local["models"])


@pytest.mark.parametrize(
    "event,message",
    [
        (None, "JSON object"),
        ({}, "missing request fields"),
        ({**_event(), "extra": True}, "unexpected request fields"),
        ({**_event(), "horizons_s": "0.25"}, "JSON array"),
        ({**_event(), "observations": "bad"}, "JSON array"),
    ],
)
def test_invalid_events_return_structured_errors(event, message):
    response = handler(event, Context())
    assert response["status"] == "error"
    assert message in response["error"]
    assert response["engine"] == "aws_lambda"


def test_future_observation_is_rejected_by_shared_engine():
    event = _event()
    event["cutoff_time_s"] = 0.95
    response = handler(event, Context())
    assert response["status"] == "error"
    assert "after cutoff_time_s" in response["error"]


def test_mocked_aws_invocation_has_exact_parity_and_measured_metadata():
    client = FakeLambda()
    result = execute_prediction(
        "AWS", _observations(), 1.1, [0.25, 0.5], client=client,
        function_name="test-function", region="ap-southeast-1",
    )

    assert result["status"] == "ok"
    assert result["engine"] == "aws_lambda"
    assert result["parity"]["passed"] is True
    assert result["parity"]["max_absolute_difference"] == 0
    assert result["metadata"]["lambda_duration_ms"] == pytest.approx(1.23)
    assert result["metadata"]["lambda_billed_duration_ms"] == pytest.approx(2.0)
    assert result["metadata"]["invocation_latency_ms"] >= 0
    assert result["local_latency_ms"] >= 0
    assert result["success_count"] == 1 and result["failure_count"] == 0
    assert client.calls[0]["InvocationType"] == "RequestResponse"


def test_aws_errors_are_visible_without_local_fallback():
    result = execute_prediction(
        "AWS", _observations(), 1.1, [0.25], client=FakeLambda("Unhandled"),
        function_name="test-function", region="ap-southeast-1",
    )
    assert result["status"] == "error"
    assert result["engine"] == "aws_lambda"
    assert "boom" in result["error"]
    assert "prediction" not in result
    assert result["success_count"] == 0 and result["failure_count"] == 1


def test_packaged_zip_imports_shared_physics(tmp_path):
    package = tmp_path / "lambda.zip"
    subprocess.run(
        ["bash", "scripts/deploy_lambda.sh", "package", str(package)],
        check=True, capture_output=True, text=True,
    )
    with zipfile.ZipFile(package) as archive:
        assert set(archive.namelist()) == {
            "lambda_handler.py", "physicseye/__init__.py", "physicseye/physics.py"
        }
