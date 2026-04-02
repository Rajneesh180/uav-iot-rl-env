# FastAPI app entrypoint
import os

from openenv.core.env_server.http_server import create_app

try:
    from ..models import UAVAction, UAVObservation
    from .uav_iot_environment import UAVIoTEnvironment
except ImportError:
    from models import UAVAction, UAVObservation
    from server.uav_iot_environment import UAVIoTEnvironment

app = create_app(UAVIoTEnvironment, UAVAction, UAVObservation, env_name="uav_iot_env")


def main():
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
