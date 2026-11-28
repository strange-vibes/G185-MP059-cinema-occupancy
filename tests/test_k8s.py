from pathlib import Path

import yaml

from app import main as app_main

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_PATH = REPO_ROOT / "k8s" / "deployment.yaml"
SERVICE_PATH = REPO_ROOT / "k8s" / "service.yaml"
DOCKERFILE_PATH = REPO_ROOT / "Dockerfile"


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_k8s_deployment_manifest_configuration():
    deployment = load_yaml(DEPLOYMENT_PATH)

    assert deployment["apiVersion"] == "apps/v1"
    assert deployment["kind"] == "Deployment"
    assert deployment["metadata"]["name"] == "mp059-api"

    spec = deployment["spec"]
    assert spec["replicas"] == 3
    pod_labels = spec["template"]["metadata"]["labels"]
    assert spec["selector"]["matchLabels"] == pod_labels
    assert pod_labels["app"] == "mp059-api"

    containers = spec["template"]["spec"]["containers"]
    assert len(containers) == 1
    container = containers[0]

    assert container["name"] == "cinema-api"
    assert container["image"] == "mp059-api:1.0"
    assert container["imagePullPolicy"] == "IfNotPresent"
    assert [p["containerPort"] for p in container["ports"]] == [8000]

    for probe_name in ("readinessProbe", "livenessProbe"):
        probe = container[probe_name]
        assert probe["httpGet"]["path"] == "/health"
        assert probe["httpGet"]["port"] == 8000

    resources = container["resources"]
    assert resources["requests"]["cpu"] == "100m"
    assert resources["requests"]["memory"] == "256Mi"
    assert resources["limits"]["cpu"] == "500m"
    assert resources["limits"]["memory"] == "512Mi"


def test_k8s_service_matches_deployment():
    deployment = load_yaml(DEPLOYMENT_PATH)
    service = load_yaml(SERVICE_PATH)

    assert service["apiVersion"] == "v1"
    assert service["kind"] == "Service"
    assert service["metadata"]["name"] == "mp059-svc"
    assert service["spec"]["type"] == "ClusterIP"

    pod_labels = deployment["spec"]["template"]["metadata"]["labels"]
    assert service["spec"]["selector"] == pod_labels

    container_port = deployment["spec"]["template"]["spec"]["containers"][0]["ports"][0][
        "containerPort"
    ]
    ports = service["spec"]["ports"]
    assert len(ports) == 1
    assert ports[0]["port"] == 80
    assert ports[0]["targetPort"] == container_port
    assert ports[0]["protocol"] == "TCP"


def test_k8s_and_dockerfile_match_fastapi_app():
    dockerfile_text = DOCKERFILE_PATH.read_text(encoding="utf-8")

    assert "EXPOSE 8000" in dockerfile_text
    assert 'CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]' in dockerfile_text
    assert "COPY app/ ./app/" in dockerfile_text
    assert "COPY models/model.joblib ./models/model.joblib" in dockerfile_text

    route_paths = {route.path for route in app_main.app.routes}
    assert "/health" in route_paths
    assert "/predict" in route_paths
    assert app_main.MODEL_PATH.exists()
