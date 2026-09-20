#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  exec sudo "$0" "$@"
fi

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${OMNI_MODEL_DIR:-$(cd "${project_dir}/../omni_ai_model" && pwd)}"
venv_dir="/opt/omni-ai-controller/venv"
config_dir="/etc/omni-ai-controller"
config_file="${config_dir}/service.env"
unit_file="/etc/systemd/system/omni-ai-controller.service"
build_dir="$(mktemp -d)"
trap 'rm -rf "${build_dir}"' EXIT

python3 -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --quiet --upgrade pip
cp "${project_dir}/pyproject.toml" "${project_dir}/README.md" "${build_dir}/"
cp -a "${project_dir}/omni_ai_controller" "${build_dir}/"
"${venv_dir}/bin/python" -m pip install --quiet "${build_dir}[service]"

install -d -m 0700 "${config_dir}"
if [[ ! -f "${config_file}" ]]; then
  token="$("${venv_dir}/bin/python" -c 'import secrets; print(secrets.token_urlsafe(32))')"
  umask 077
  cat >"${config_file}" <<EOF
OMNI_MODEL_DIR=${model_dir}
OMNI_ADMIN_TOKEN=${token}
OMNI_ALLOWED_NETWORKS=192.168.192.0/24
OMNI_ALLOWED_CONTAINERS=omni-ai-model,omni-ai-receipt-ocr,omni-ai-main-service,omni-ai-database
OMNI_CONTROLLER_SOCKET=/run/omni-ai-controller/controller.sock
EOF
elif grep -q '^OMNI_ALLOWED_CONTAINERS=' "${config_file}" && \
     ! grep -q '^OMNI_ALLOWED_CONTAINERS=.*omni-ai-receipt-ocr' "${config_file}"; then
  sed -i '/^OMNI_ALLOWED_CONTAINERS=/ s/$/,omni-ai-receipt-ocr/' "${config_file}"
fi
chmod 0600 "${config_file}"
install -d -m 0755 /run/omni-ai-controller

sed \
  -e "s|@PROJECT_DIR@|${project_dir}|g" \
  -e "s|@MODEL_DIR@|${model_dir}|g" \
  "${project_dir}/deploy/omni-ai-controller.service" >"${unit_file}"
chmod 0644 "${unit_file}"

systemctl daemon-reload
systemctl enable --now omni-ai-controller.service
systemctl --no-pager --full status omni-ai-controller.service

printf '\nAdmin token is stored in %s (mode 0600).\n' "${config_file}"
printf 'For security it is not printed by this installer.\n'
