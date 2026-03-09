#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Launcher paths.
CONFIG_FILE="${CONFIG_FILE:-accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-training/train_dynin_omni_stage1.py}"
EXPERIMENT_CFG="${EXPERIMENT_CFG:-configs/dynin_omni_stage1_llada_instruct.yaml}"
LOG_DIR="${LOG_DIR:-logs}"

# Distributed runtime.
MAIN_PORT="${MAIN_PORT:-8888}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-}"
HOSTS_CSV="${HOSTS_CSV:-localhost}"
SSH_PORT="${SSH_PORT:-22}"
NCCL_DEBUG_LEVEL="${NCCL_DEBUG_LEVEL:-INFO}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PROJECT_ROOT}/.cache/triton}"
TORCH_EXTENSIONS_ROOT="${TORCH_EXTENSIONS_ROOT:-${PROJECT_ROOT}/.cache/torch_extensions}"
TMP_ROOT="${TMP_ROOT:-${PROJECT_ROOT}/.cache/tmp}"
CPU_ADAM_REQUIRED_GLIBCXX="${CPU_ADAM_REQUIRED_GLIBCXX:-GLIBCXX_3.4.32}"
ENFORCE_MAIN_HOST_LAUNCH="${ENFORCE_MAIN_HOST_LAUNCH:-1}"
ALLOW_NON_MAIN_HOST_LAUNCH="${ALLOW_NON_MAIN_HOST_LAUNCH:-0}"

# Env activation on each rank.
# - If REMOTE_SETUP is set, it is used as-is (highest priority).
# - Otherwise this script initializes conda and, when CONDA_ENV_NAME is non-empty,
#   activates that env before launch.
REMOTE_SETUP="${REMOTE_SETUP:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dynin}"
CONDA_SH_PATH="${CONDA_SH_PATH:-}"

validate_project_relative_path() {
  local input_path="$1"
  local var_name="$2"

  if [[ -z "${input_path}" ]]; then
    echo "${var_name} must not be empty." >&2
    exit 1
  fi
  if [[ "${input_path}" == /* || "${input_path}" == "~"* ]]; then
    echo "${var_name} must be a project-root-relative path (not absolute): ${input_path}" >&2
    exit 1
  fi
  if [[ "${input_path}" =~ (^|/)\.\.(/|$) ]]; then
    echo "${var_name} must stay inside project root (no '..'): ${input_path}" >&2
    exit 1
  fi
}

resolve_project_relative_path() {
  local input_path="$1"
  local var_name="$2"
  validate_project_relative_path "${input_path}" "${var_name}"
  printf "%s/%s" "${PROJECT_ROOT}" "${input_path}"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [--help]

Launch multi-stage omni-modal training with accelerate.

Environment variables:
  CONFIG_FILE       Accelerate config YAML (project-relative). Default: accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml
  TRAIN_SCRIPT      Python entrypoint (project-relative). Default: training/train_dynin_omni_stage1.py
  EXPERIMENT_CFG    Training config (project-relative). Default: configs/dynin_omni_stage1_llada_instruct.yaml
  LOG_DIR           Log root directory (project-relative). A run_YYYYMMDD_HHMMSS folder is created under this path. Default: logs
  HOSTS_CSV         Comma-separated host list in rank order. Default: localhost
  MAIN_PROCESS_IP   Master IP for accelerate. Default: first host in HOSTS_CSV
  MAIN_PORT         Master port for accelerate. Default: 8888
  SSH_PORT          SSH port for remote ranks. Default: 22
  NCCL_DEBUG_LEVEL  NCCL_DEBUG value. Default: INFO
  TRITON_CACHE_DIR  Triton autotune cache directory. Default: <project>/.cache/triton
  TORCH_EXTENSIONS_ROOT  Root for host-separated torch extension caches.
  TMP_ROOT          Root for host-separated temporary directories (TMPDIR/TMP/TEMP).
  CPU_ADAM_REQUIRED_GLIBCXX  Required symbol for CPUAdam when optimizer offload is cpu.
  ENFORCE_MAIN_HOST_LAUNCH  If 1, abort when launched from a non-main host.
  HF_DATASETS_OFFLINE  Passed to all ranks if set.
  HF_HUB_OFFLINE       Passed to all ranks if set.
  TRANSFORMERS_OFFLINE Passed to all ranks if set.

  REMOTE_SETUP      Optional shell snippet executed before launch on each host.
                    Example: source /path/to/conda.sh && conda activate myenv && export LD_LIBRARY_PATH=...
  CONDA_SH_PATH     Optional path to conda.sh used when REMOTE_SETUP is empty.
  CONDA_ENV_NAME    Conda env to activate when REMOTE_SETUP is empty.
                    Set empty (CONDA_ENV_NAME=) to skip activation.

Examples:
  # Default run
  ./scripts/train.sh

  # Custom remote setup (recommended for deployment)
  CONDA_ENV_NAME=dynin \\
  EXPERIMENT_CFG=configs/dynin_omni_stage1_llada_instruct.yaml \\
  TRAIN_SCRIPT=training/train_dynin_omni_stage1.py \\
  ./scripts/train.sh
EOF
}

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

CONFIG_FILE="$(resolve_project_relative_path "${CONFIG_FILE}" "CONFIG_FILE")"
TRAIN_SCRIPT="$(resolve_project_relative_path "${TRAIN_SCRIPT}" "TRAIN_SCRIPT")"
EXPERIMENT_CFG="$(resolve_project_relative_path "${EXPERIMENT_CFG}" "EXPERIMENT_CFG")"
LOG_DIR="$(resolve_project_relative_path "${LOG_DIR}" "LOG_DIR")"
TRAIN_NAME="$(basename "${TRAIN_SCRIPT}")"
TRAIN_NAME="${TRAIN_NAME%.*}"
TRAIN_NAME_SAFE="${TRAIN_NAME//[^A-Za-z0-9_.-]/_}"

build_setup_cmd() {
  if [[ -n "${REMOTE_SETUP}" ]]; then
    printf "%s" "${REMOTE_SETUP}"
    return
  fi

  local -a setup_parts=()
  if [[ -n "${CONDA_SH_PATH}" ]]; then
    setup_parts+=("source ${CONDA_SH_PATH}")
  fi

  if [[ -n "${CONDA_ENV_NAME}" ]]; then
    setup_parts+=("conda activate ${CONDA_ENV_NAME}")
  fi

  local setup_cmd=""
  local part
  for part in "${setup_parts[@]}"; do
    setup_cmd="${setup_cmd:+${setup_cmd} && }${part}"
  done

  printf "%s" "${setup_cmd}"
}

is_local_host() {
  local host="$1"
  local local_ipv4
  local_ipv4="$(hostname -I 2>/dev/null || true)"

  [[ "$host" == "localhost" \
    || "$host" == "$(hostname)" \
    || "$host" == "$(hostname -f)" \
    || " ${local_ipv4} " == *" ${host} "* ]]
}

declare -a HOSTS=()
IFS=',' read -r -a RAW_HOSTS <<< "${HOSTS_CSV}"
for raw_host in "${RAW_HOSTS[@]}"; do
  host="${raw_host#"${raw_host%%[![:space:]]*}"}"
  host="${host%"${host##*[![:space:]]}"}"
  [[ -n "${host}" ]] || continue
  HOSTS+=("${host}")
done

if (( ${#HOSTS[@]} == 0 )); then
  echo "HOSTS_CSV is empty after parsing; set at least one host." >&2
  exit 1
fi

if [[ -z "${MAIN_PROCESS_IP}" ]]; then
  MAIN_PROCESS_IP="${HOSTS[0]}"
fi

if (( ${#HOSTS[@]} > 1 )) && [[ "${ENFORCE_MAIN_HOST_LAUNCH}" == "1" ]] && [[ "${ALLOW_NON_MAIN_HOST_LAUNCH}" != "1" ]]; then
  if ! is_local_host "${MAIN_PROCESS_IP}"; then
    echo "Refusing to launch from a non-main host." >&2
    echo "Run this script only on the main node (${MAIN_PROCESS_IP}) or set ALLOW_NON_MAIN_HOST_LAUNCH=1." >&2
    exit 1
  fi
fi

for required_file in "${CONFIG_FILE}" "${TRAIN_SCRIPT}" "${EXPERIMENT_CFG}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Missing required file: ${required_file}" >&2
    exit 1
  fi
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${LOG_DIR}/run_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

declare -a PIDS=()
declare -a HOST_LABELS=()

timestamp_lines() {
  while IFS= read -r line; do
    printf "%s %s\n" "$(date "+%Y-%m-%d %H:%M:%S")" "$line"
  done
}

stop_all() {
  if (( ${#PIDS[@]} == 0 )); then
    return
  fi
  echo "Stopping launched processes..."
  for pid in "${PIDS[@]}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" >/dev/null 2>&1 || kill "${pid}" >/dev/null 2>&1 || true
    fi
  done

  sleep 2
  for pid in "${PIDS[@]}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" >/dev/null 2>&1 || kill -KILL "${pid}" >/dev/null 2>&1 || true
    fi
  done
}

on_signal() {
  echo "Signal received, terminating all ranks."
  stop_all
  exit 1
}
trap on_signal INT TERM

supports_wait_n() {
  help wait 2>/dev/null | grep -q -- '-n'
}

launch_rank() {
  local host="$1"
  local rank="$2"
  local log_file="$3"
  local host_label="$4"
  local base_cmd
  local config_file_escaped
  local cpu_adam_precheck_cmd
  local launch_cmd
  local lib_setup_cmd
  local python_guard_cmd
  local runtime_env
  local setup_cmd
  local tmp_setup_cmd
  local tmp_dir
  local env_scope
  local triton_cache_dir
  local torch_extensions_dir

  setup_cmd="$(build_setup_cmd)"
  env_scope="${CONDA_ENV_NAME:-noenv}"
  tmp_dir="${TMP_ROOT}/${env_scope}/${host_label}"
  triton_cache_dir="${TRITON_CACHE_DIR}/${env_scope}/${host_label}"
  torch_extensions_dir="${TORCH_EXTENSIONS_ROOT}/${env_scope}/${host_label}"
  config_file_escaped="$(printf "%q" "${CONFIG_FILE}")"

  lib_setup_cmd='unset LD_PRELOAD; if [[ -n "${CONDA_PREFIX:-}" ]]; then export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"; fi'

  cpu_adam_precheck_cmd="if grep -Eq '^[[:space:]]*offload_optimizer_device:[[:space:]]*cpu([[:space:]]|$)' ${config_file_escaped}; then if [[ -z \"\${CONDA_PREFIX:-}\" ]]; then echo 'CONDA_PREFIX is empty while CPU optimizer offload is enabled.' >&2; exit 1; fi; if [[ ! -f \"\${CONDA_PREFIX}/lib/libstdc++.so.6\" ]]; then echo \"Missing \${CONDA_PREFIX}/lib/libstdc++.so.6 required for CPUAdam.\" >&2; exit 1; fi; if ! strings \"\${CONDA_PREFIX}/lib/libstdc++.so.6\" | grep -q '${CPU_ADAM_REQUIRED_GLIBCXX}'; then echo \"\${CONDA_PREFIX}/lib/libstdc++.so.6 lacks ${CPU_ADAM_REQUIRED_GLIBCXX}; CPUAdam will fail.\" >&2; exit 1; fi; fi"
  python_guard_cmd="if [[ -n \"\${CONDA_PREFIX:-}\" ]]; then py_path=\$(python -c 'import sys; print(sys.executable)'); case \"\${py_path}\" in \"\${CONDA_PREFIX}/bin/\"*) ;; *) echo \"python executable (\${py_path}) is outside CONDA_PREFIX (\${CONDA_PREFIX}).\" >&2; exit 1 ;; esac; fi; python -c 'import accelerate.commands' >/dev/null"
  tmp_setup_cmd="mkdir -p ${tmp_dir} ${triton_cache_dir} ${torch_extensions_dir} && export TMPDIR=${tmp_dir} TMP=${tmp_dir} TEMP=${tmp_dir}"

  runtime_env="PYTHONNOUSERSITE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python NCCL_DEBUG=${NCCL_DEBUG_LEVEL} NCCL_SHM_DISABLE=1 NCCL_ASYNC_ERROR_HANDLING=1 TRITON_CACHE_DIR=${triton_cache_dir} TORCH_EXTENSIONS_DIR=${torch_extensions_dir} TMPDIR=${tmp_dir} TMP=${tmp_dir} TEMP=${tmp_dir}"
  if [[ -n "${HF_DATASETS_OFFLINE:-}" ]]; then
    runtime_env="${runtime_env} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE}"
  fi
  if [[ -n "${HF_HUB_OFFLINE:-}" ]]; then
    runtime_env="${runtime_env} HF_HUB_OFFLINE=${HF_HUB_OFFLINE}"
  fi
  if [[ -n "${TRANSFORMERS_OFFLINE:-}" ]]; then
    runtime_env="${runtime_env} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE}"
  fi
  launch_cmd="env ${runtime_env} python -m accelerate.commands.launch --config_file ${CONFIG_FILE} --num_machines ${#HOSTS[@]} --machine_rank ${rank} --main_process_ip ${MAIN_PROCESS_IP} --main_process_port ${MAIN_PORT} ${TRAIN_SCRIPT} config=${EXPERIMENT_CFG}"
  base_cmd="${setup_cmd} && ${lib_setup_cmd} && ${tmp_setup_cmd} && ${python_guard_cmd} && ${cpu_adam_precheck_cmd} && cd ${PROJECT_ROOT} && ${launch_cmd}"

  if is_local_host "$host"; then
    echo "[rank ${rank}] running locally (${host_label}), logging to ${log_file}"
    setsid bash -lc "${base_cmd}" > >(timestamp_lines >"${log_file}") 2>&1 &
  else
    local dest="${SSH_USER:-$USER}@${host}"
    local ssh_opts="${SSH_OPTS:--o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$HOME/.ssh/known_hosts}"
    local escaped_cmd
    escaped_cmd=$(printf "%q" "${base_cmd}")
    echo "[rank ${rank}] ssh ${dest}, logging to ${log_file}"
    setsid ssh -p "${SSH_PORT}" ${ssh_opts} "${dest}" "bash -lc ${escaped_cmd}" > >(timestamp_lines >"${log_file}") 2>&1 &
  fi

  PIDS[${rank}]=$!
  HOST_LABELS[${rank}]="${host_label}"
}

for idx in "${!HOSTS[@]}"; do
  host="${HOSTS[$idx]}"
  safe_host=${host//[^A-Za-z0-9_.-]/_}
  log_file="${LOG_DIR}/${TRAIN_NAME_SAFE}_${TIMESTAMP}_rank${idx}_${safe_host}.log"
  launch_rank "${host}" "${idx}" "${log_file}" "${safe_host}"
done

echo "All nodes launched. Tail logs under ${LOG_DIR}."

if supports_wait_n; then
  active_ranks=()
  for rank in "${!PIDS[@]}"; do
    [[ -n "${PIDS[$rank]:-}" ]] || continue
    active_ranks+=("${rank}")
  done

  while (( ${#active_ranks[@]} > 0 )); do
    active_pids=()
    for rank in "${active_ranks[@]}"; do
      pid="${PIDS[$rank]}"
      [[ -n "${pid:-}" ]] || continue
      active_pids+=("${pid}")
    done

    (( ${#active_pids[@]} > 0 )) || break

    if wait -n "${active_pids[@]}"; then
      status=0
    else
      status=$?
    fi

    failed_rank=""
    failed_status=0
    next_active_ranks=()
    for rank in "${active_ranks[@]}"; do
      pid="${PIDS[$rank]}"
      if [[ -n "${pid:-}" ]] && kill -0 "${pid}" 2>/dev/null; then
        next_active_ranks+=("${rank}")
      else
        rank_status=127
        if [[ -n "${pid:-}" ]]; then
          if wait "${pid}"; then
            rank_status=0
          else
            rank_status=$?
          fi
        fi
        if [[ -z "${failed_rank}" ]] && (( rank_status != 0 && rank_status != 127 )); then
          failed_rank="${rank}"
          failed_status="${rank_status}"
        fi
        PIDS[$rank]=""
      fi
    done
    active_ranks=("${next_active_ranks[@]}")

    if (( status != 0 || failed_status != 0 )); then
      if [[ -n "${failed_rank}" ]]; then
        echo "[rank ${failed_rank}] (${HOST_LABELS[$failed_rank]}) exited with status ${failed_status}"
      elif (( status != 0 )); then
        echo "A rank process exited with status ${status}"
      else
        echo "A rank process exited with a non-zero status."
      fi
      stop_all
      if (( failed_status != 0 )); then
        exit "${failed_status}"
      fi
      exit "${status}"
    fi
  done
else
  for rank in "${!PIDS[@]}"; do
    pid="${PIDS[$rank]}"
    [[ -n "${pid:-}" ]] || continue
    if wait "${pid}"; then
      continue
    else
      status=$?
      echo "[rank ${rank}] (${HOST_LABELS[$rank]}) exited with status ${status}"
      stop_all
      exit "${status}"
    fi
  done
fi
