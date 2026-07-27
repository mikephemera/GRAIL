#!/bin/bash
# Install / augment the `sonic` conda env used by GRAIL retargeting + SONIC training.
#
# Usage:
#   bash scripts/setup/install_env_sonic.sh              # default env 'sonic'
#   GRAIL_SONIC_ENV=my_sonic_env bash scripts/setup/install_env_sonic.sh
#   BOOTSTRAP_SONIC=0 bash scripts/setup/install_env_sonic.sh         # retarget-only / mjlab
#                                                                      (skip Isaac Sim/Lab)
#   GRAIL_GMR_DIR=/workspace/GMR_musa bash scripts/setup/install_env_sonic.sh
#   RETARGET_ONLY=0 BOOTSTRAP_SONIC=0 bash scripts/setup/install_env_sonic.sh # existing sonic env
#   INSTALL_SYSTEM_DEPS=0 bash scripts/setup/install_env_sonic.sh      # skip apt step
#   PULL_LFS=0 bash scripts/setup/install_env_sonic.sh                 # skip git-lfs pull
#
# What this script does, in order:
#   -1. (INSTALL_SYSTEM_DEPS=1 — default for full SONIC, 0 for retarget-only)
#       Install vulkan/GUI libs + git-lfs via apt. Uses sudo if needed;
#       no-op if we're neither root nor have sudo.
#   0. (BOOTSTRAP_SONIC=1 — default) Create the conda env with Python 3.11,
#      pip-install Isaac Sim 5.1.0 (`isaacsim[all,extscache]`), clone Isaac
#      Lab v2.3.2 to $ISAAC_LAB_DIR (default: ~/IsaacLab), run
#      `./isaaclab.sh --install all`, pip install the core `isaaclab`
#      editable, and install `vector_quantize_pytorch`. Set BOOTSTRAP_SONIC=0
#      for a MUSA/MuJoCo-only stageD retarget runtime; this defaults to
#      RETARGET_ONLY=1 unless overridden.
#   1. Resolves a GMR checkout from either `imports/GMR` or `GRAIL_GMR_DIR`
#      (for a fork such as `/workspace/GMR_musa`), then applies NVIDIA GMR
#      overrides from grail/retargeting/gmr_overrides/ on top of it.
#   2. Full SONIC only: symlinks data/motion_lib_genhoi + models into
#      imports/SONIC/gear_sonic/.
#   3. pip install -e ${GMR_DIR} + GRAIL package (editable). Retarget-only
#      installs GMR with --no-deps and lets the retargeting dependency block
#      below provide PyPI runtime deps. Full SONIC also installs
#      imports/SONIC/gear_sonic[training] + huggingface_hub.
#   4. pip install retargeting-specific deps (mujoco, pxr, trimesh, ...).
#   5. Sanity-imports the top-level modules.
#   6. (PULL_LFS=1 — default when git-lfs is on PATH) git-lfs pull on
#      imports/SONIC so the robot mesh STLs + policy ONNX materialize.

set -eo pipefail

ENV_NAME="${GRAIL_SONIC_ENV:-sonic}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OVERRIDES="${REPO_ROOT}/grail/retargeting/gmr_overrides"
BOOTSTRAP_SONIC="${BOOTSTRAP_SONIC:-1}"
if [[ -z "${RETARGET_ONLY+x}" ]]; then
    if [[ "${BOOTSTRAP_SONIC}" == "0" ]]; then
        RETARGET_ONLY=1
    else
        RETARGET_ONLY=0
    fi
fi
if [[ -z "${INSTALL_SYSTEM_DEPS+x}" ]]; then
    if [[ "${RETARGET_ONLY}" == "1" ]]; then
        INSTALL_SYSTEM_DEPS=0
    else
        INSTALL_SYSTEM_DEPS=1
    fi
fi
if [[ -z "${PULL_LFS+x}" ]]; then
    if [[ "${RETARGET_ONLY}" == "1" ]]; then
        PULL_LFS=0
    else
        PULL_LFS=1
    fi
fi
ISAAC_LAB_DIR="${ISAAC_LAB_DIR:-$HOME/IsaacLab}"
ISAAC_SIM_VERSION="${ISAAC_SIM_VERSION:-5.1.0}"
ISAAC_LAB_TAG="${ISAAC_LAB_TAG:-v2.3.2}"

echo ">>> Target conda env: ${ENV_NAME}"
echo ">>> Repo root:        ${REPO_ROOT}"
echo ">>> Bootstrap mode:   ${BOOTSTRAP_SONIC} (1=install Isaac Sim/Lab, 0=retarget-only)"
echo ">>> Retarget-only:    ${RETARGET_ONLY} (1=skip SONIC training deps)"
echo ">>> System deps:      ${INSTALL_SYSTEM_DEPS} (1=apt install, 0=skip apt)"

# --- Step -1: system deps (Vulkan/GUI/git-lfs) via apt ------------------
# Idempotent: re-installs are a fast pass. Skipped entirely on non-apt
# systems or when we can't elevate.
if [[ "${INSTALL_SYSTEM_DEPS}" == "1" ]] && command -v apt-get &>/dev/null; then
    APT_PKGS=(
        libvulkan1 vulkan-tools mesa-vulkan-drivers
        libxcb-xfixes0 libxcb-cursor0 libxrandr2 libxi6 libxcursor1
        libxtst6 libxss1 libxrender1 libgl1 libegl1
        git-lfs rsync
    )
    if [[ "$(id -u)" -eq 0 ]]; then
        APT_CMD="apt-get"
    elif sudo -n true 2>/dev/null; then
        APT_CMD="sudo apt-get"
    else
        APT_CMD=""
        echo ">>> [skip apt] not root and no passwordless sudo; install these manually if missing:"
        echo "    ${APT_PKGS[*]}"
    fi
    if [[ -n "${APT_CMD}" ]]; then
        echo ">>> Installing system deps via ${APT_CMD} (Vulkan, GUI, git-lfs)"
        ${APT_CMD} update -qq
        ${APT_CMD} install -y --no-install-recommends "${APT_PKGS[@]}" | tail -3
    fi
fi

# --- Step 0: bootstrap or select Python env -----------------------------
if command -v conda &>/dev/null; then
    eval "$(conda shell.bash hook)"
else
    CONDA_CANDIDATES=(
        "/root/miniconda3/etc/profile.d/conda.sh"
        "/root/anaconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
    )
    for conda_sh in "${CONDA_CANDIDATES[@]}"; do
        if [[ -f "${conda_sh}" ]]; then
            # shellcheck disable=SC1090
            source "${conda_sh}"
            break
        fi
    done
fi

if ! command -v conda &>/dev/null; then
    if [[ "${RETARGET_ONLY}" == "1" ]]; then
        echo ">>> [skip conda] conda not found; installing into active Python: $(python -c 'import sys; print(sys.executable)')"
    else
        echo "ERROR: conda not found; full SONIC setup requires conda." >&2
        exit 1
    fi
elif [[ "${BOOTSTRAP_SONIC}" == "1" ]]; then
    if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        echo ">>> Creating conda env '${ENV_NAME}' with Python 3.11"
        conda create -y -n "${ENV_NAME}" python=3.11
    fi
    conda activate "${ENV_NAME}"

    echo ">>> Upgrading pip"
    pip install --upgrade pip

    if ! python -c "import isaacsim" 2>/dev/null; then
        echo ">>> Installing Isaac Sim ${ISAAC_SIM_VERSION} (~6 GB download)"
        pip install "isaacsim[all,extscache]==${ISAAC_SIM_VERSION}" \
            --extra-index-url https://pypi.nvidia.com
    fi

    # Accept EULA non-interactively on first import. The Kit kernel checks
    # for the literal file <isaacsim_pkg>/kit/EULA_ACCEPTED before showing
    # its interactive prompt — write it directly so this works in non-TTY
    # builds (CI, Docker image bakes) where stdin is closed and the
    # `python -c "import isaacsim"` workaround silently fails.
    export OMNI_KIT_ACCEPT_EULA=Yes
    ISAACSIM_PKG=$(python -c "import isaacsim, os; print(os.path.dirname(isaacsim.__file__))")
    echo "yes" > "${ISAACSIM_PKG}/kit/EULA_ACCEPTED"
    python -c "import isaacsim" >/dev/null

    # Pre-install flatdict without build isolation. flatdict 4.0.1 (pinned by
    # Isaac Lab core) has a legacy setup.py that imports pkg_resources, which
    # setuptools 81+ removed. PEP 517 build isolation installs the latest
    # setuptools, so the wheel build fails. Pin setuptools<81 in the env
    # first, then build flatdict against it.
    pip install 'setuptools<81' wheel
    pip install 'flatdict==4.0.1' --no-build-isolation

    if [[ ! -d "${ISAAC_LAB_DIR}" ]]; then
        echo ">>> Cloning Isaac Lab ${ISAAC_LAB_TAG} to ${ISAAC_LAB_DIR}"
        git clone --depth 1 --branch "${ISAAC_LAB_TAG}" \
            https://github.com/isaac-sim/IsaacLab.git "${ISAAC_LAB_DIR}"
    fi

    if ! python -c "import isaaclab" 2>/dev/null; then
        echo ">>> Running ./isaaclab.sh --install all (~10-15 min, ~4 GB)"
        (cd "${ISAAC_LAB_DIR}" && ./isaaclab.sh --install all)
        # Isaac Lab's --install flag sometimes skips the core `isaaclab`
        # package when a transitive dep (e.g., flatdict) failed during the
        # first pass. Install it explicitly to be safe.
        pip install -e "${ISAAC_LAB_DIR}/source/isaaclab"
    fi

    # Required by some gear_sonic configs.
    pip install vector_quantize_pytorch
else
    if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        if [[ "${RETARGET_ONLY}" == "1" ]]; then
            echo ">>> Creating retarget-only conda env '${ENV_NAME}' with Python 3.11"
            conda create -y -n "${ENV_NAME}" python=3.11
        else
            echo "ERROR: conda env '${ENV_NAME}' does not exist and BOOTSTRAP_SONIC=0." >&2
            echo "       Either unset BOOTSTRAP_SONIC (default bootstrap) or create the env first." >&2
            exit 1
        fi
    fi
    conda activate "${ENV_NAME}"
fi

DEFAULT_GMR_DIR="${REPO_ROOT}/imports/GMR"
EXTERNAL_GMR_DIR="$(cd "${REPO_ROOT}/.." && pwd)/GMR_musa"
GMR_DIR="${GRAIL_GMR_DIR:-${DEFAULT_GMR_DIR}}"
if [[ ! -d "${GMR_DIR}/general_motion_retargeting" ]]; then
    if [[ -z "${GRAIL_GMR_DIR:-}" ]] && [[ -d "${EXTERNAL_GMR_DIR}/general_motion_retargeting" ]]; then
        GMR_DIR="${EXTERNAL_GMR_DIR}"
    fi
fi
if [[ ! -d "${GMR_DIR}/general_motion_retargeting" ]]; then
    echo "ERROR: no GMR checkout found." >&2
    echo "       Expected ${DEFAULT_GMR_DIR} or set GRAIL_GMR_DIR=/path/to/GMR_fork" >&2
    exit 1
fi
echo ">>> GMR checkout:    ${GMR_DIR}"

# --- Step 1: apply NVIDIA GMR overrides ---------------------------------
# GRAIL's GMR customizations are applied at runtime via grail/adapters/gmr.py
# (monkey-patching) and data/g1_smplx/gmr_smplx_to_g1.json (IK config).
# The gmr_overrides/ directory provided file-based overrides in an older
# approach; if it exists, apply it, otherwise skip.
if [[ -d "${OVERRIDES}" ]]; then
    echo ">>> Applying GMR overrides: ${OVERRIDES} -> ${GMR_DIR}"
    rsync -a --exclude='README.md' "${OVERRIDES}/" "${GMR_DIR}/"
else
    echo ">>> [skip] No gmr_overrides/ directory — using runtime adapter patches instead"
fi

GEAR_SONIC="${REPO_ROOT}/imports/SONIC/gear_sonic"
if [[ "${RETARGET_ONLY}" != "1" ]]; then
    # --- Step 2: surface data/ and models/ into the SONIC submodule -----
    # imports/SONIC/gear_sonic/ is the cwd for training scripts; it expects
    # data/motion_lib_genhoi/... and models/... to resolve from there.
    mkdir -p "${REPO_ROOT}/data/motion_lib_genhoi" "${REPO_ROOT}/models"
    ln -sfn ../../../../data/motion_lib_genhoi "${GEAR_SONIC}/data/motion_lib_genhoi"
    ln -sfn ../../../models "${GEAR_SONIC}/models"
    echo ">>> Linked ${GEAR_SONIC}/{data/motion_lib_genhoi,models} -> repo root"
else
    echo ">>> [skip] SONIC data/model symlinks (retarget-only)"
fi

# --- Step 3: editable installs ------------------------------------------
if [[ "${RETARGET_ONLY}" == "1" ]]; then
    echo ">>> pip install --no-deps -e ${GMR_DIR} (retarget-only)"
    pip install --no-deps -e "${GMR_DIR}"
else
    echo ">>> pip install -e ${GMR_DIR}"
    pip install -e "${GMR_DIR}"
fi

if [[ "${RETARGET_ONLY}" != "1" ]]; then
    echo ">>> pip install -e imports/SONIC/gear_sonic[training] + huggingface_hub"
    pip install -e "${GEAR_SONIC}[training]"
    pip install huggingface_hub
else
    echo ">>> [skip] imports/SONIC/gear_sonic[training] (retarget-only)"
fi

echo ">>> pip install -e . (grail, --no-deps)"
# --no-deps: grail's setup.cfg has unpinned numpy/opencv-python, which resolve
# to numpy 2.x + opencv 4.13 and break gear_sonic (numpy==1.26.4), isaaclab-rl
# (numpy<2), and isaacsim-kernel (numpy==1.26.0). The sonic env only consumes
# grail.retargeting; its real deps are installed by GMR and the retargeting
# dependency block below.
pip install --no-deps -e "${REPO_ROOT}"

# --- Step 4: retargeting-specific deps ----------------------------------
echo ">>> pip install retargeting deps"
RETARGET_DEPS=(
    'numpy<2'
    loop_rate_limiters
    joblib
    mink
    mujoco
    natsort
    'opencv-python<4.12'
    psutil
    protobuf
    'qpsolvers[proxqp]'
    'redis[hiredis]'
    'imageio[ffmpeg]'
    smplx
    trimesh
    usd-core
    scipy
    rich
    tqdm
)
if [[ "${RETARGET_ONLY}" != "1" ]]; then
    RETARGET_DEPS+=(
        'simple-raycaster @ git+https://github.com/Agent-3154/simple-raycaster.git@197daa6dcb146c5ce3e675a173328e17df6b9777'
    )
fi
pip install "${RETARGET_DEPS[@]}"

if [[ "${RETARGET_ONLY}" != "1" ]]; then
    # --- Step 4b: SONIC training/eval-callback deps ---------------------
    # smpl_sim is a non-PyPI package providing compute_metrics_lite, used by
    # the SONIC eval-watcher's im_eval callback.
    pip install \
        numpy-stl easydict gymnasium mediapy torchgeometry vtk \
        'smpl_sim @ git+https://github.com/ZhengyiLuo/SMPLSim.git'
else
    echo ">>> [skip] SONIC training/eval deps (retarget-only)"
fi

# --- Step 6: git-lfs pull for SONIC assets ------------------------------
# Mesh STLs + policy ONNX files are LFS-tracked. Without this pull, the
# preflight check fails at the size check (pointer files are <1 KB).
if [[ "${PULL_LFS}" == "1" ]] && command -v git-lfs &>/dev/null; then
    echo ">>> git lfs install + pull in imports/SONIC"
    # Avoid git's "dubious ownership" refusal when running as root inside a
    # container against a bind-mounted host repo (different UIDs).
    git config --global --add safe.directory "${REPO_ROOT}" 2>/dev/null || true
    git config --global --add safe.directory "${REPO_ROOT}/imports/SONIC" 2>/dev/null || true
    # --skip-repo: set up global LFS filters only. Without it, `git lfs install`
    # aborts with exit 2 when an identical pre-push hook already exists in the
    # cwd repo (idempotency foot-gun under `set -e`). `git lfs pull` below
    # works regardless since SONIC already has the hook.
    git lfs install --skip-repo
    (cd "${REPO_ROOT}/imports/SONIC" && git lfs pull) | tail -3 || \
        echo "  [WARN] git lfs pull failed — run manually: cd imports/SONIC && git lfs pull"
elif [[ "${PULL_LFS}" == "1" ]]; then
    echo ">>> [skip git-lfs pull] git-lfs not on PATH — install it and run:"
    echo "    cd imports/SONIC && git lfs pull"
fi

# --- Sanity checks -------------------------------------------------------
echo ">>> Verifying install"
python -c "import general_motion_retargeting as gmr; print(f'  GMR: {gmr.__file__}')"
python -c "from grail.retargeting.retarget import main; print('  grail.retargeting.retarget: OK')"
python -c "import smplx, mujoco; print('  smplx, mujoco: OK')"
if [[ "${RETARGET_ONLY}" != "1" && "${BOOTSTRAP_SONIC}" == "1" ]]; then
    OMNI_KIT_ACCEPT_EULA=Yes python -c "import isaaclab, isaacsim; print('  isaaclab + isaacsim: OK')"
fi

echo ""
echo "Setup complete. Quick start:"
echo "  bash grail/retargeting/scripts/retarget_pipeline.sh <data_dir> <output_folder>"
if [[ "${RETARGET_ONLY}" != "1" ]]; then
    echo ""
    echo "Full preflight:"
    echo "  OMNI_KIT_ACCEPT_EULA=Yes python imports/SONIC/check_environment.py --training"
fi
