"""Never read the deployed model .env while importing the global service app."""
from pathlib import Path
from unittest.mock import patch

from omni_ai_controller.config import ServerConfig

# The real service creates its global app at import time. Supply an inert config
# only for that import; individual tests retain their own explicit fake settings.
with patch.object(ServerConfig, "from_model_dir", return_value=ServerConfig(
    model_dir=Path("/tmp/controller-offline-test-model"), base_url="http://127.0.0.1:1",
    api_key="offline-test", control_token="offline-test", model_name="offline-test",
)):
    import omni_ai_controller.service  # noqa: F401