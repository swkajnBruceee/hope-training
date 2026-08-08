#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
package_dir=$(cd "$script_dir/.." && pwd)
source "$package_dir/env.sh"

start_iox_roudi() {
  if pgrep -f "$script_dir/iox-roudi" >/dev/null; then
    echo "iox-roudi is already running"
  else
    "$script_dir/iox-roudi" &
    echo "iox-roudi started"
  fi
}

select_cfg() {
  local cfg_directory="$script_dir/cfg"
  local requested=${1:-}

  if [[ -n "$requested" ]]; then
    if [[ -f "$requested" ]]; then
      realpath "$requested"
      return
    fi
    if [[ -f "$cfg_directory/$requested" ]]; then
      realpath "$cfg_directory/$requested"
      return
    fi
    echo "Cfg not found: $requested" >&2
    exit 1
  fi

  mapfile -t cfg_files < <(find "$cfg_directory" -maxdepth 1 -mindepth 1 -type f -name '*.yaml' | sort)
  if ((${#cfg_files[@]} == 0)); then
    echo "No cfg files found in $cfg_directory" >&2
    exit 1
  fi

  echo "Please select a cfg file:"
  for i in "${!cfg_files[@]}"; do
    printf "%s: %s\n" "$i" "$(basename "${cfg_files[$i]}")"
  done

  local choice
  read -r -p "Please enter your choice: " choice
  if [[ "$choice" =~ ^[0-9]+$ && "$choice" -ge 0 && "$choice" -lt "${#cfg_files[@]}" ]]; then
    realpath "${cfg_files[$choice]}"
  else
    echo "Invalid choice. Exiting." >&2
    exit 1
  fi
}

start_iox_roudi
cd "$script_dir"
cfg_file=$(select_cfg "${1:-}")
echo "Selected cfg file: $cfg_file"
exec "$script_dir/aimrt_main" --cfg_file_path="$cfg_file"
