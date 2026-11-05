#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .tools logs cache models runs datasets cache/libero
export UV_CACHE_DIR="$PWD/cache/uv" UV_PYTHON_INSTALL_DIR="$PWD/.tools/python" UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=2 MODELSCOPE_CACHE="$PWD/cache/modelscope"
uv="$PWD/.tools/uv-x86_64-unknown-linux-gnu/uv"
if [[ ! -x "$uv" ]];then
 curl -L --fail https://github.com/astral-sh/uv/releases/download/0.9.7/uv-x86_64-unknown-linux-gnu.tar.gz -o .tools/uv.tar.gz
 tar -xzf .tools/uv.tar.gz -C .tools
fi
"$uv" python install 3.11
if [[ ! -d vendor/openpi/.git ]];then
 git clone https://github.com/Physical-Intelligence/openpi.git vendor/openpi
 git -C vendor/openpi checkout 215abfb217dbac7d5f1273282331b9b1866c0479
fi
if [[ "$(git -C vendor/openpi rev-parse HEAD)" != 215abfb217dbac7d5f1273282331b9b1866c0479 ]];then echo 'Unexpected openpi revision; preserving existing checkout, stopping setup.';exit 1;fi
"$uv" venv --allow-existing --python /usr/bin/python3.10 .venv-tools
"$uv" pip install --python .venv-tools/bin/python modelscope==1.40.2 requests==2.34.2 tqdm==4.70.1 google-crc32c==1.7.1
"$uv" venv --allow-existing --python /usr/bin/python3.10 .venv-sim
"$uv" pip sync --python .venv-sim/bin/python deployment/sim-requirements.lock
"$uv" pip install --python .venv-sim/bin/python --no-deps -e ./vendor/openpi/packages/openpi-client
"$uv" sync --project vendor/openpi --python 3.11 --frozen --no-dev
python3 - <<'PY'
import pathlib
p=pathlib.Path.cwd();a=p/'LIBERO/libero/libero'
values={'benchmark_root':a,'bddl_files':a/'bddl_files','init_states':a/'init_files','datasets':p/'datasets','assets':a/'assets'}
(p/'cache/libero/config.yaml').write_text('\n'.join(f'{k}: {v}' for k,v in values.items())+'\n')
PY
.venv-tools/bin/python scripts/download_pi05.py
.venv-tools/bin/python scripts/verify_checkpoint.py
.venv-tools/bin/python - <<'PY'
from modelscope.hub.snapshot_download import dataset_snapshot_download
print(dataset_snapshot_download('ataier/LIBERO_Recovery_FailureScenes', revision='95acf98dd41a4965018069e7e4f27aac1fc69328',local_dir='benchmark_failure',allow_file_pattern=['index/*','scenes/*','annotations/scenes/*'],max_workers=2))
PY
mkdir -p cache/openpi/big_vision
if [[ ! -s cache/openpi/big_vision/paligemma_tokenizer.model ]];then curl -L --fail https://storage.googleapis.com/big_vision/paligemma_tokenizer.model -o cache/openpi/big_vision/paligemma_tokenizer.model;fi
python3 scripts/status.py --write
