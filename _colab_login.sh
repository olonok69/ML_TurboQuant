#!/bin/bash
set -euo pipefail
export CLOUDSDK_CONFIG="${HOME}/.config/colab-gcloud"
mkdir -p "$CLOUDSDK_CONFIG"
# Open the consent page in the Windows browser when WSL has no display.
if ! command -v wslview >/dev/null 2>&1; then
  export BROWSER="cmd.exe /c start"
fi
gcloud auth application-default login \
  --scopes="openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory"
echo "ADC_READY"
