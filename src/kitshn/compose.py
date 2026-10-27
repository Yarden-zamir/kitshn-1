from __future__ import annotations

from dataclasses import dataclass
import json
import time
from pathlib import Path
from typing import Any, Iterable

from .builders import HOST_BUILDER, ensure_builder
from .errors import KitshnError
from .filesystem import reset_socket_paths, roll_service_logs, walk_deployments
from .models import Deployment, Recipe, Roots
from .preview_limits import PreviewLimits, apply_preview_limits, preview_builder
from .runner import CommandRunner

# The file names that Compose finds on its own, in its order of preference (compose-go
# DefaultFileNames and DefaultOverrideFileNames). Revisit if Compose changes them.
COMPOSE_FILE_NAMES = ("compose.yaml", "compose.yml", "docker-compose.yml", "docker-compose.yaml")
COMPOSE_OVERRIDE_NAMES = (
    "compose.override.yml",
    "compose.override.yaml",
    "docker-compose.override.yml",
    "docker-compose.override.yaml",
)


@dataclass(frozen=True, slots=True)
class ComposeService:
    name: str
    labels: dict[str, str]
    has_build: bool
    image: str | None
    pull_policy: str | None
    has_healthcheck: bool
    networks: list[str]


def has_compose_file(deployment: Deployment) -> bool:
    return (deployment.deployment_root / "compose.yml").exists() or (
        deployment.deployment_root / "compose.yaml"
    ).exists()


def render_compose_config(deployment: Deployment, runner: CommandRunner) -> dict[str, Any]:
    if not has_compose_file(deployment):
        return {"services": {}}
    result = runner.run(
        compose_command(deployment, "config", "--format", "json"),
        cwd=deployment.deployment_root,
        env=compose_env(deployment),
        capture=True,
    )
    try:
        config = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        msg = "docker compose config did not return valid JSON"
        raise KitshnError(msg) from error
    if not isinstance(config, dict):
        msg = "docker compose config JSON must be an object"
        raise KitshnError(msg)
    return config


def compose_services(config: dict[str, Any]) -> list[ComposeService]:
    raw_services = config.get("services", {})
    if not isinstance(raw_services, dict):
        msg = "compose config services must be an object"
        raise KitshnError(msg)

    services: list[ComposeService] = []
    for name, raw_service in raw_services.items():
        if not isinstance(name, str) or not isinstance(raw_service, dict):
            msg = "compose service entries must be named objects"
            raise KitshnError(msg)
        services.append(
            ComposeService(
                name=name,
                labels=_labels_as_dict(raw_service.get("labels", {})),
                has_build="build" in raw_service,
                image=raw_service.get("image") if isinstance(raw_service.get("image"), str) else None,
                pull_policy=(
                    raw_service.get("pull_policy")
                    if isinstance(raw_service.get("pull_policy"), str)
                    else None
                ),
                has_healthcheck="healthcheck" in raw_service,
                networks=_networks_as_list(raw_service.get("networks", [])),
            )
        )
    return services


def ignored_dotenv_warning(recipe_dir: Path) -> str | None:
    """Explain that a recipe `.env` has no effect, or None when the recipe has none.

    KitSHn passes `--env-file <params.env>` to every Compose command, so Compose never reads
    `.env`. Values and COMPOSE_PROFILES in it silently do nothing.
    """

    dotenv = recipe_dir / ".env"
    if not dotenv.is_file():
        return None
    keys: list[str] = []
    for raw_line in dotenv.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            key = line.split("=", 1)[0].removeprefix("export ").strip()
            # Continuation lines of multi-line quoted values are not keys.
            if key.isidentifier():
                keys.append(key)
    names = ", ".join(keys) if keys else "no keys"
    return (
        f".env in the recipe is not used for interpolation or COMPOSE_PROFILES ({names}). "
        "KitSHn runs Compose with --env-file params.env. A service env_file entry still reads it. "
        "Use GitHub vars or secrets named KITSHN_<NAME>, or compose.override.yml."
    )


def apply_compose(
    deployment: Deployment, runner: CommandRunner, limits: PreviewLimits | None = None
) -> list[str]:
    # The override of the last deploy names that deploy's services. Remove it first: the config
    # below is then the recipe's own, and a deploy without limits runs without them.
    if not runner.dry_run:
        deployment.preview_limits_file.unlink(missing_ok=True)
    config = render_compose_config(deployment, runner)
    services = compose_services(config)
    if not services:
        reset_socket_paths(deployment)
        return []

    service_names = [service.name for service in services]
    if limits is not None:
        apply_preview_limits(deployment, service_names, limits, runner)
    external_services = [service for service in services if service.image and not service.has_build]
    build_services = [service for service in services if service.has_build]

    if external_services:
        runner.run(
            compose_command(deployment, "pull", "--ignore-buildable", *[s.name for s in external_services]),
            cwd=deployment.deployment_root,
            env=compose_env(deployment),
        )
    if build_services:
        if limits is None:
            ensure_builder(HOST_BUILDER, runner)
        runner.run(
            compose_command(deployment, "build", *[s.name for s in build_services]),
            cwd=deployment.deployment_root,
            env=compose_env(deployment),
        )

    # Clear sockets only once images are ready: a failed pull or build keeps the running
    # containers and their live socket serving.
    reset_socket_paths(deployment)
    roll_service_logs(deployment.logs_root, service_names)
    runner.run(
        compose_command(deployment, "up", "-d", "--remove-orphans", "--force-recreate"),
        cwd=deployment.deployment_root,
        env=compose_env(deployment),
    )
    wait_for_healthchecks(deployment, services, runner)
    return service_names


def recreate_dependent_services(
    changed_recipe: Recipe, current: Deployment, roots: Roots, runner: CommandRunner
) -> list[str]:
    changed: list[str] = []
    for deployment in walk_deployments(roots):
        if deployment.identity == current.identity or not has_compose_file(deployment):
            continue
        if not deployment.params_file.exists():
            continue
        config = render_compose_config(deployment, runner)
        services = compose_services(config)
        dependent = [
            service for service in services if _depends_on_recipe(service, changed_recipe.full_name)
        ]
        if not dependent:
            continue
        names = [service.name for service in dependent]
        roll_service_logs(deployment.logs_root, names)
        runner.run(
            compose_command(deployment, "up", "-d", "--no-deps", "--force-recreate", *names),
            cwd=deployment.deployment_root,
            env=compose_env(deployment),
        )
        wait_for_healthchecks(deployment, dependent, runner)
        changed.extend(f"{deployment.identity}:{name}" for name in names)
    return changed


def compose_down(deployment: Deployment, runner: CommandRunner, *, volumes: bool = False) -> None:
    """Remove the project's containers, networks, and locally built images, by project name.

    Compose finds them by their project label, so a checkout that is gone, or whose compose file
    changed since the last deploy, still tears down fully. It runs from the deployments root,
    where there is no compose file, so Compose loads none and needs no params. Locally built
    images are rebuilt on the next deploy; pulled images stay in the cache.
    """

    args = ["docker", "compose", "--project-name", deployment.compose_project, "down", "--remove-orphans", "--rmi", "local"]
    if volumes:
        args.append("--volumes")
    deployment.roots.deployments.mkdir(parents=True, exist_ok=True)
    runner.run(args, cwd=deployment.roots.deployments)
    if runner.dry_run:
        return
    # A destroy that reports success must leave no container behind, so check by the project label.
    left = runner.run(
        ["docker", "ps", "--all", "--quiet", "--filter", f"label=com.docker.compose.project={deployment.compose_project}"],
        capture=True,
    ).stdout.split()
    if left:
        msg = f"compose down left {len(left)} container(s) of {deployment.compose_project}: {' '.join(left)}"
        raise KitshnError(msg)


def compose_service_status(deployment: Deployment, runner: CommandRunner) -> list[dict[str, Any]]:
    if not has_compose_file(deployment):
        return []
    result = runner.run(
        compose_command(deployment, "ps", "--format", "json"),
        cwd=deployment.deployment_root,
        env=compose_env(deployment),
        capture=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return _parse_compose_ps_json(result.stdout)


def run_compose_command(
    deployment: Deployment,
    args: list[str],
    runner: CommandRunner,
) -> None:
    if not has_compose_file(deployment):
        msg = f"deployment has no compose file: {deployment.identity}"
        raise KitshnError(msg)
    if not deployment.params_file.exists():
        msg = f"deployment has no params file: {deployment.params_file}"
        raise KitshnError(msg)
    if not args:
        msg = "compose command requires docker compose arguments, for example: kitshn compose owner/repo -- ps"
        raise KitshnError(msg)
    runner.run(compose_command(deployment, *args), cwd=deployment.deployment_root, env=compose_env(deployment))


def wait_for_healthchecks(
    deployment: Deployment,
    services: Iterable[ComposeService],
    runner: CommandRunner,
    *,
    timeout_seconds: int = 120,
) -> None:
    healthchecked = [service for service in services if service.has_healthcheck]
    if not healthchecked:
        return

    deadline = time.monotonic() + timeout_seconds
    pending = {service.name for service in healthchecked}
    while pending and time.monotonic() < deadline:
        for service_name in list(pending):
            status = _service_health(deployment, service_name, runner)
            if status == "healthy":
                pending.remove(service_name)
            elif status == "unhealthy":
                msg = f"service became unhealthy: {deployment.identity}:{service_name}"
                raise KitshnError(msg)
        if pending:
            time.sleep(2)

    if pending:
        msg = "timed out waiting for healthy services: " + ", ".join(sorted(pending))
        raise KitshnError(msg)


def _service_health(deployment: Deployment, service_name: str, runner: CommandRunner) -> str | None:
    ps = runner.run(
        compose_command(deployment, "ps", "-q", service_name),
        cwd=deployment.deployment_root,
        env=compose_env(deployment),
        capture=True,
    )
    container_ids = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
    if not container_ids:
        msg = f"no container found for healthchecked service: {deployment.identity}:{service_name}"
        raise KitshnError(msg)
    inspect = runner.run(
        ["docker", "inspect", "--format", "{{if .State.Health}}{{.State.Health.Status}}{{end}}", container_ids[0]],
        capture=True,
    )
    return inspect.stdout.strip() or None


def compose_env(deployment: Deployment) -> dict[str, str]:
    """The environment of every Compose command of a deployment.

    A deployment without limits builds in the host builder. A limited deployment adds its
    override to the files that Compose finds on its own, and selects the recipe's preview builder. So a later command, such as an `up` that rebuilds an
    image with `pull_policy: build`, keeps the limits too.
    """

    if not deployment.preview_limits_file.exists():
        return {**deployment.runtime_env, "BUILDX_BUILDER": HOST_BUILDER}
    root = deployment.deployment_root
    base = [root / name for name in COMPOSE_FILE_NAMES if (root / name).exists()][:1]
    override = [root / name for name in COMPOSE_OVERRIDE_NAMES if (root / name).exists()][:1]
    if not base:
        msg = f"deployment has preview limits but no compose file: {deployment.identity}"
        raise KitshnError(msg)
    files = [*base, *override, deployment.preview_limits_file]
    return {
        **deployment.runtime_env,
        "COMPOSE_FILE": ":".join(str(path) for path in files),
        "BUILDX_BUILDER": preview_builder(deployment.recipe),
    }


def compose_command(deployment: Deployment, *args: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        deployment.compose_project,
        "--env-file",
        str(deployment.params_file),
        *args,
    ]


def _labels_as_dict(raw_labels: Any) -> dict[str, str]:
    if isinstance(raw_labels, dict):
        return {str(key): str(value) for key, value in raw_labels.items()}
    if isinstance(raw_labels, list):
        labels: dict[str, str] = {}
        for raw_label in raw_labels:
            if not isinstance(raw_label, str) or "=" not in raw_label:
                continue
            key, value = raw_label.split("=", 1)
            labels[key] = value
        return labels
    if raw_labels in (None, []):
        return {}
    msg = "compose labels must be a mapping or list"
    raise KitshnError(msg)


def _networks_as_list(raw_networks: Any) -> list[str]:
    if isinstance(raw_networks, dict):
        return [str(key) for key in raw_networks]
    if isinstance(raw_networks, list):
        return [str(value) for value in raw_networks]
    if raw_networks in (None, []):
        return []
    msg = "compose networks must be a mapping or list"
    raise KitshnError(msg)


def _depends_on_recipe(service: ComposeService, recipe_name: str) -> bool:
    raw = service.labels.get("kitshn.depends_on")
    if raw is None:
        return False
    normalized_recipe_name = recipe_name.lower()
    return normalized_recipe_name in [part.strip().lower() for part in raw.split(",") if part.strip()]


def _parse_compose_ps_json(output: str) -> list[dict[str, Any]]:
    stripped = output.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = [json.loads(line) for line in stripped.splitlines() if line.strip()]
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
        return parsed
    msg = "docker compose ps JSON had an unexpected shape"
    raise KitshnError(msg)
