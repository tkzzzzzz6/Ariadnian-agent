"""Keep offline tests independent of a developer's real provider configuration."""
import os

# Individual tests still exercise environment/UI overrides with monkeypatch.
# Loading a local .env here could override their expected defaults or credentials.
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
