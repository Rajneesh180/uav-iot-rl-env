"""WebSocket client for the UAV-IoT environment."""

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from .models import UAVAction, UAVObservation, UAVState


class UAVIoTEnv(EnvClient[UAVAction, UAVObservation, UAVState]):
    """WebSocket client wrapping the UAV-IoT environment server."""

    def _step_payload(self, action: UAVAction) -> dict:
        """Convert UAVAction to JSON payload for step request."""
        return {"action": action.action}

    def _parse_result(self, payload: dict) -> StepResult[UAVObservation]:
        """Parse server response into StepResult[UAVObservation]."""
        obs = UAVObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> UAVState:
        """Parse server response into UAVState object."""
        return UAVState(**payload)
