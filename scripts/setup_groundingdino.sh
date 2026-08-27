#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
grounding_dir="${repo_root}/GroundingDINO"
grounding_commit="856dde20aee659246248e20734ef9ba5214f5e44"
compat_patch="${repo_root}/patches/groundingdino-compat.patch"

if [[ ! -d "${grounding_dir}/.git" ]]; then
  git clone https://github.com/IDEA-Research/GroundingDINO.git "${grounding_dir}"
fi

if [[ -n "$(git -C "${grounding_dir}" status --porcelain)" ]]; then
  echo "GroundingDINO contains local changes; refusing to overwrite them." >&2
  exit 1
fi

git -C "${grounding_dir}" fetch origin "${grounding_commit}"
git -C "${grounding_dir}" checkout --detach "${grounding_commit}"

if git -C "${grounding_dir}" apply --check "${compat_patch}"; then
  git -C "${grounding_dir}" apply "${compat_patch}"
elif git -C "${grounding_dir}" apply --reverse --check "${compat_patch}"; then
  echo "GroundingDINO compatibility patch is already applied."
else
  echo "GroundingDINO compatibility patch does not apply cleanly." >&2
  exit 1
fi

echo "Place groundingdino_swinb_cogcoor.pth in ${grounding_dir}/"
