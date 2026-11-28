"""Human-run OAuth repair using the installed Colab CLI's official login flow.

Run in a local terminal with the colab-cli Python interpreter. Authorization
codes are entered only into that terminal. No token contents are printed.
Existing credentials are replaced only after successful authorization.
"""
from pathlib import Path
import os
import tempfile


def main():
    from colab_cli import auth
    target = Path(auth.TOKEN_CONFIG_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="colab-reauth-", dir=target.parent) as directory:
        temporary = Path(directory) / "token.json"
        auth.TOKEN_CONFIG_PATH = str(temporary)
        config = str(Path.home() / ".colab-cli-oauth-config.json")
        credentials = auth._get_google_auth_credentials(config)
        if not credentials.valid or not temporary.is_file():
            raise RuntimeError("Authorization did not produce valid credentials")
        os.replace(temporary, target)
    print("Colab login renewed. No GPU session was created.")


if __name__ == "__main__":
    main()
