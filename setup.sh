#!/usr/bin/env bash
# Rampart Image Studio setup: checks the GPU, finds rampart (or installs it into ./rampart) and the
# rampart modules Rampart Image Studio needs, installs the engine's Python packages into engine/pylib
# (with rampart's pip3r) and downloads the model weights into weights/.  Safe to re-run:
# finished steps are skipped, interrupted downloads resume.
#
#   ./setup.sh               everything for this machine
#   ./setup.sh --fp8         also fetch the FP8 transformer (for NVIDIA GPUs the INT8 kernels don't support)
#   ./setup.sh --verify      re-check the SHA-256 of every downloaded file
#   ./setup.sh --no-python   weights only;  --no-weights: packages only
#   ./setup.sh --own-rampart   install rampart into ./rampart even if one is found elsewhere
#   ./setup.sh --port 8088 --host 127.0.0.1   the web address, without asking (setup.sh asks for
#                            both when run in a terminal; 0.0.0.0 listens on every interface)
#   ./setup.sh --engine-gpu 1 --enhancer-gpu 0   with several NVIDIA GPUs, which runs what, without
#                            asking (the enhancer may be split across several: --enhancer-gpu 0,2)
#   RAMPART=/path/to/bin/rampart ./setup.sh    use that rampart
#   RAMPART_INSTALLER=URL ./setup.sh           fetch rampart's installer script from URL
#                                              (default https://get.rampart.dev/)
#
# Weights come from public Hugging Face mirrors of Black Forest Labs' release and are
# checked against BFL's SHA-256 (weights.manifest).  If every mirror of a file is gone,
# set HF_TOKEN to a token whose account accepted the FLUX Kommunity License to fetch it
# from black-forest-labs directly.  The weights are under that license; see README.md.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
# compiled bytecode goes into the project, never next to rampart's python (as engine/engine.sh)
export PYTHONPYCACHEPREFIX=$ROOT/engine/pycache

PYTHON=1 WEIGHTS=1 VERIFY=0 FP8=0 OWN_RAMPART=0 OPT_PORT= OPT_HOST= OPT_ENGINE_GPU= OPT_ENHANCER_GPU=
while [ $# -gt 0 ]; do
    case $1 in
        --no-python)  PYTHON=0 ;;
        --no-weights) WEIGHTS=0 ;;
        --verify)     VERIFY=1 ;;
        --fp8)        FP8=1 ;;
        --own-rampart) OWN_RAMPART=1 ;;
        --port=*)     OPT_PORT=${1#*=} ;;
        --host=*)     OPT_HOST=${1#*=} ;;
        --engine-gpu=*)   OPT_ENGINE_GPU=${1#*=} ;;
        --enhancer-gpu=*) OPT_ENHANCER_GPU=${1#*=} ;;
        --port|--host|--engine-gpu|--enhancer-gpu)
            [ $# -gt 1 ] || { echo "$1 needs a value (see --help)" >&2; exit 1; }
            case $1 in
                --port) OPT_PORT=$2 ;; --host) OPT_HOST=$2 ;;
                --engine-gpu) OPT_ENGINE_GPU=$2 ;; --enhancer-gpu) OPT_ENHANCER_GPU=$2 ;;
            esac; shift ;;
        -h|--help)    sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)            echo "unknown option: $1 (see --help)" >&2; exit 1 ;;
    esac
    shift
done

say() { printf '\n== %s\n' "$*"; }
die() { echo "setup: $*" >&2; exit 1; }
ver_ge() {   # dotted version $1 >= $2
    awk -v a="$1" -v b="$2" 'BEGIN { split(a, x, "."); split(b, y, ".")
        for (i = 1; i <= 3; i++) { if (x[i] + 0 > y[i] + 0) exit 0; if (x[i] + 0 < y[i] + 0) exit 1 }
        exit 0 }'
}
resolve() {   # a path with every symlink followed (readlink -f is missing on older macOS)
    local p=$1 t
    while [ -L "$p" ]; do
        t=$(readlink "$p")
        case $t in /*) p=$t ;; *) p=$(dirname "$p")/$t ;; esac
    done
    echo "$(cd "$(dirname "$p")" && pwd -P)/$(basename "$p")"
}
conf_get() {   # a setting from studio.conf (empty if unset)
    sed -n "s/^$1=\([^ #]*\).*/\1/p" studio.conf 2>/dev/null | tail -1 | tr -d '"' || true
}
conf_set() {   # KEY VALUE [comment]: change KEY in studio.conf, keeping its comment, or add it
    if grep -q "^$1=" studio.conf; then
        awk -v k="$1" -v v="$2" 'index($0, k "=") == 1 { sub("^" k "=[^ #]*", k "=" v) } { print }' \
            studio.conf > studio.conf.tmp && mv studio.conf.tmp studio.conf
    else
        printf '%s=%s%s\n' "$1" "$2" "${3:+   # $3}" >> studio.conf
    fi
}
TMPS=()
trap 'rm -rf ${TMPS[@]+"${TMPS[@]}"}' EXIT

# --- tools --------------------------------------------------------------------------
# curl downloads rampart's installer and the weights (macOS always has it; minimal Linux
# installs, containers and older Ubuntu desktops may not)
command -v curl >/dev/null || die "curl is needed to download rampart and the model weights, and it is not installed.
       Install it, then run ./setup.sh again:
           sudo apt install curl     (Debian, Ubuntu)
           sudo dnf install curl     (Fedora, RHEL)"

# --- platform and GPU (before anything is installed) -------------------------------
# The engine's PyTorch and the prompt enhancer bring their own CUDA libraries, so a
# Linux machine needs only the NVIDIA driver (580 series or newer, for CUDA 13), not
# the CUDA toolkit.  nvidia-smi comes with the driver's utilities and can be missing
# where the driver itself works (compute-only installs, containers), so it is not the test.
MIN_DRIVER=580
case "$(uname -s)/$(uname -m)" in
    Linux/*)
        PLATFORM=linux driver=
        if [ -r /proc/driver/nvidia/version ]; then
            driver=$(grep -oE ' [0-9]{3,}\.[0-9]+(\.[0-9]+)? ' /proc/driver/nvidia/version | head -1 | tr -d ' ' || true)
        elif ! { PATH=$PATH:/sbin:/usr/sbin ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1'; }; then
            die "no NVIDIA driver found (the Linux engine needs an NVIDIA GPU with driver $MIN_DRIVER or newer)"
        fi
        [ -z "$driver" ] && command -v nvidia-smi >/dev/null \
            && driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
        if [ -z "$driver" ]; then
            echo "warning: NVIDIA driver found but its version is unknown; Rampart Image Studio needs $MIN_DRIVER or newer"
        elif ! ver_ge "$driver" "$MIN_DRIVER"; then
            die "NVIDIA driver $driver is too old; Rampart Image Studio needs $MIN_DRIVER or newer (CUDA 13)"
        fi
        gpus=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | paste -sd ';' - || true)
        echo "NVIDIA driver ${driver:-?}${gpus:+; GPUs: $gpus}" ;;
    Darwin/arm64) PLATFORM=mac ;;
    *) die "unsupported platform $(uname -s)/$(uname -m): Linux with an NVIDIA GPU, or an Apple silicon Mac" ;;
esac

# --- web address ------------------------------------------------------------------
# asked for in a terminal (the current settings offered as the defaults), else kept as they
# are or set by --port/--host; a rerun changes only what differs
[ -f studio.conf ] || { cp studio.conf.example studio.conf; say "Created studio.conf (ports, GPUs): edit as needed"; }
valid_port() { [[ $1 =~ ^[0-9]+$ ]] && [ "$1" -ge 1 ] && [ "$1" -le 65535 ]; }
valid_host() {   # an IPv4 address (localhost: 127.0.0.1)
    local IFS=. o
    [[ $1 =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || return 1
    for o in $1; do [ "$o" -le 255 ] || return 1; done
}
ask() {   # QUESTION DEFAULT CHECK: the answer, asked again until CHECK accepts it
    local a
    while :; do
        read -r -p "$1 [$2]: " a || a=
        a=${a:-$2}; [ "$a" = localhost ] && a=127.0.0.1
        "$3" "$a" && { echo "$a"; return; }
        echo "  not valid: $a" >&2
    done
}
cur_port=$(conf_get WEB_PORT) cur_host=$(conf_get HOST)
if [ -z "$cur_host" ]; then   # settings from before HOST: BIND_ALL=1 listened on every interface
    if [ "$(conf_get BIND_ALL)" = 1 ]; then cur_host=0.0.0.0; else cur_host=127.0.0.1; fi
fi
cur_port=${cur_port:-8088}
[ "$OPT_HOST" = localhost ] && OPT_HOST=127.0.0.1
[ -z "$OPT_PORT" ] || valid_port "$OPT_PORT" || die "--port $OPT_PORT: give a number from 1 to 65535"
[ -z "$OPT_HOST" ] || valid_host "$OPT_HOST" || die "--host $OPT_HOST: give an IPv4 address (0.0.0.0: every interface)"
port=$OPT_PORT host=$OPT_HOST
if [ -t 0 ] && [ -z "$port$host" ]; then
    say "Web address (Enter keeps the value in brackets)"
    port=$(ask "Port for Rampart Image Studio" "$cur_port" valid_port)
    host=$(ask "Address to listen on (127.0.0.1: this machine only, 0.0.0.0: every interface)" "$cur_host" valid_host)
fi
port=${port:-$cur_port} host=${host:-$cur_host}
url="http://$host:$port/studio/"; [ "$host" = 0.0.0.0 ] && url="http://$(hostname):$port/studio/ (every interface)"
if [ "$port" != "$(conf_get WEB_PORT)" ] || [ "$host" != "$(conf_get HOST)" ] || grep -q '^BIND_ALL=' studio.conf; then
    conf_set WEB_PORT "$port" "Rampart Image Studio at http://<host>:WEB_PORT/studio/"
    # settings from before HOST: its line takes BIND_ALL's place
    note="address to listen on: 127.0.0.1 this machine only, 0.0.0.0 every interface"
    if ! grep -q '^HOST=' studio.conf; then
        awk -v h="HOST=$host" -v n="$note" '/^BIND_ALL=/ && !done { printf "%-22s # %s\n", h, n; done = 1; next } { print }' \
            studio.conf > studio.conf.tmp && mv studio.conf.tmp studio.conf
    fi
    conf_set HOST "$host" "$note"
    awk '!/^BIND_ALL=/' studio.conf > studio.conf.tmp && mv studio.conf.tmp studio.conf
    echo "Web address: $url, saved in studio.conf"
    if { [ "$port" != "$cur_port" ] || [ "$host" != "$cur_host" ]; } && [ -f web_server/server.pid ] \
        && kill -0 "$(cat web_server/server.pid)" 2>/dev/null; then
        echo "Rampart Image Studio is running on the old address; restart it to use the new one: ./studio.sh restart"
    fi
else
    echo "Web address: $url, unchanged"
fi

# --- GPUs ---------------------------------------------------------------------------
# With more than one NVIDIA GPU: which one runs the image engine (it uses one) and which
# the prompt enhancer (one, or several: llama.cpp splits its layers across them).  Asked in
# a terminal like the web address, with the current settings as the defaults, or on a first
# run a suggestion; otherwise set by --engine-gpu/--enhancer-gpu, or the suggestion when unset.
gpu_list=
[ $PLATFORM = linux ] && command -v nvidia-smi >/dev/null \
    && gpu_list=$(nvidia-smi --query-gpu=index,memory.total,compute_cap,name --format=csv,noheader,nounits 2>/dev/null || true)
ngpu=$(printf '%s\n' "$gpu_list" | grep -c . || true)
if [ "$ngpu" -gt 1 ]; then
    # "index GB compute_cap" per GPU, the engine's choice first: the most memory (to the GB,
    # so two 12 GB cards count as equal), then the newest architecture
    ranked=$(printf '%s\n' "$gpu_list" | awk -F', *' '{ printf "%s %d %s\n", $1, ($2 + 0) / 1024 + 0.5, $3 + 0 }' | sort -k2,2nr -k3,3gr)
    gpu_ids=" $(printf '%s\n' "$gpu_list" | cut -d, -f1 | tr '\n' ' ')"
    rec_engine=$(echo "$ranked" | head -1 | cut -d' ' -f1)
    # the enhancer needs about 6.3 GB (an 8 GB card): the largest other GPU that holds it, or the fewest other
    # GPUs that do together, or else the engine's GPU (the engine then takes what is left)
    rec_enhancer=$(echo "$ranked" | tail -n +2 | awk -v e="$rec_engine" '
        { gb += $2; ids = ids (ids == "" ? "" : ",") $1; if (gb >= 8) { print ids; found = 1; exit } }
        END { if (!found) print e }')
    valid_gpu() { [[ $1 =~ ^[0-9]+$ ]] && [[ $gpu_ids == *" $1 "* ]]; }
    valid_gpus() {   # one GPU, or several separated by commas, each once
        local IFS=, g seen=" "
        [[ $1 =~ ^[0-9]+(,[0-9]+)*$ ]] || return 1
        for g in $1; do valid_gpu "$g" && [[ $seen != *" $g "* ]] || return 1; seen+="$g "; done
    }
    gpu_name() { local IFS=, g out=; for g in $1; do out+="${out:+ + }$(printf '%s\n' "$gpu_list" | awk -F', *' -v i="$g" '$1 == i { print $4 }')"; done; echo "$out"; }
    cur_engine=$(conf_get ENGINE_GPU) cur_enhancer=$(conf_get ENHANCER_GPU)
    valid_gpu "$cur_engine" || cur_engine=$rec_engine
    valid_gpus "$cur_enhancer" || cur_enhancer=$rec_enhancer
    [ -z "$OPT_ENGINE_GPU" ] || valid_gpu "$OPT_ENGINE_GPU" || die "--engine-gpu $OPT_ENGINE_GPU: give one GPU number from nvidia-smi ($(echo $gpu_ids))"
    [ -z "$OPT_ENHANCER_GPU" ] || valid_gpus "$OPT_ENHANCER_GPU" || die "--enhancer-gpu $OPT_ENHANCER_GPU: give one GPU number from nvidia-smi, or several like 0,1 ($(echo $gpu_ids))"
    engine_gpu=$OPT_ENGINE_GPU enhancer_gpu=$OPT_ENHANCER_GPU
    if [ -t 0 ] && [ -z "$engine_gpu$enhancer_gpu" ]; then
        say "GPUs (Enter keeps the value in brackets)"
        printf '%s\n' "$gpu_list" | awk -F', *' '{ m = $2 + 0; printf "  %s  %-30s %s\n", $1, $4, m ? sprintf("%.0f GB", m / 1024) : "" }'
        echo "The image engine runs on one GPU.  The prompt enhancer needs about 6.3 GB: on a GPU of its"
        echo "own the engine keeps all of the other, or it can be split across several (like 0,1)."
        engine_gpu=$(ask "GPU for the image engine" "$cur_engine" valid_gpu)
        enhancer_gpu=$(ask "GPU(s) for the prompt enhancer" "$cur_enhancer" valid_gpus)
    fi
    engine_gpu=${engine_gpu:-$cur_engine} enhancer_gpu=${enhancer_gpu:-$cur_enhancer}
    gpus_text="engine on $engine_gpu ($(gpu_name "$engine_gpu")), enhancer on $enhancer_gpu ($(gpu_name "$enhancer_gpu"))"
    if [ "$engine_gpu" != "$(conf_get ENGINE_GPU)" ] || [ "$enhancer_gpu" != "$(conf_get ENHANCER_GPU)" ]; then
        old_gpus="$(conf_get ENGINE_GPU)/$(conf_get ENHANCER_GPU)"
        conf_set ENGINE_GPU "$engine_gpu"
        conf_set ENHANCER_GPU "$enhancer_gpu"
        echo "GPUs: $gpus_text, saved in studio.conf"
        if [ "$old_gpus" != / ] && [ -f web_server/server.pid ] && kill -0 "$(cat web_server/server.pid)" 2>/dev/null; then
            echo "Rampart Image Studio is running on the old GPUs; restart it to use the new ones: ./studio.sh restart"
        fi
    else
        echo "GPUs: $gpus_text, unchanged"
    fi
fi

# --- rampart ------------------------------------------------------------------------
# The first rampart of at least MIN_RAMPART from: $RAMPART, the one setup.sh chose before
# (studio.conf), ./rampart, the PATH, ~/.rampart and /usr/local/rampart.  With none,
# install one for Rampart Image Studio alone into ./rampart.  Its modules come from its own tree.
MIN_RAMPART=0.7.3
RAMPART_INSTALLER=${RAMPART_INSTALLER:-https://get.rampart.dev/}
rampart_version() { "$1" --version 2>/dev/null | awk '$1 == "version:" { print $2; exit }'; }

install_rampart() {
    local tmp bundle
    tmp=$(mktemp -d); TMPS+=("$tmp")
    say "Installing rampart into $ROOT/rampart"
    curl -fsSL "$RAMPART_INSTALLER" -o "$tmp/get-rampart.sh" || die "could not download the rampart installer from $RAMPART_INSTALLER"
    (cd "$tmp" && RAMPART_NO_RUN=1 sh get-rampart.sh) || die "could not download rampart"
    bundle=$(ls "$tmp"/rampart-install-* | grep -v '\.sha1$' | head -1)
    # an installer with --prefix installs there without touching PATH, shell files or /usr/local
    if ! "$bundle" --help 2>/dev/null | grep -q -- '--prefix'; then
        cat >&2 <<EOF
setup: rampart $MIN_RAMPART or newer was not found, and this rampart installer cannot yet
       install into a directory of its own choosing.  Install rampart, then run ./setup.sh again:
           curl -fsSL https://get.rampart.dev/ | sh
       (or point setup.sh at a rampart elsewhere: RAMPART=/path/to/bin/rampart ./setup.sh)
EOF
        exit 1
    fi
    # created first: the installer asks for root for any prefix under /usr/ (as a clone in
    # /usr/local/src would be) unless the user already owns that directory
    mkdir -p "$ROOT/rampart"
    "$bundle" --auto --prefix "$ROOT/rampart" || die "the rampart installer failed"
}

RAMPART_SET=${RAMPART:-}
conf_rampart=$(conf_get RAMPART)
RAMPART= rejected=
candidates=("$RAMPART_SET" "$conf_rampart" "$ROOT/rampart/bin/rampart" "$(command -v rampart || true)"
            "$HOME/.rampart/bin/rampart" /usr/local/rampart/bin/rampart)
if [ $OWN_RAMPART = 1 ]; then
    [ -z "$RAMPART_SET" ] || die "--own-rampart and RAMPART= cannot be used together"
    candidates=("$ROOT/rampart/bin/rampart")
fi
for r in "${candidates[@]}"; do
    [ -n "$r" ] && [ -x "$r" ] || continue
    v=$(rampart_version "$r")
    if [ -n "$v" ] && ver_ge "$v" "$MIN_RAMPART"; then RAMPART=$r; break; fi
    rejected+="  $r: version ${v:-unknown}"$'\n'
done
[ -n "$RAMPART_SET" ] && [ "$RAMPART" != "$RAMPART_SET" ] && die "RAMPART=$RAMPART_SET is not rampart $MIN_RAMPART or newer"
if [ -z "$RAMPART" ]; then
    [ -n "$rejected" ] && printf 'rampart older than %s, not used:\n%s' "$MIN_RAMPART" "$rejected"
    install_rampart
    RAMPART=$ROOT/rampart/bin/rampart
fi
case $RAMPART in /*) ;; *) RAMPART=$PWD/$RAMPART ;; esac
RAMPART_HOME=$(dirname "$(dirname "$(resolve "$RAMPART")")")   # its install tree: bin/, modules/
echo "rampart $(rampart_version "$RAMPART") at $RAMPART_HOME"

# modules, by exact package name (an alias like rampart-llamacpp asks which build).  On
# Linux the enhancer needs llama.cpp's CUDA 13 build, which runs on the CUDA libraries
# installed with PyTorch into engine/pylib (studio.sh puts them on LD_LIBRARY_PATH).
modules=()
[ -x "$RAMPART_HOME/bin/python3r" ] && [ -x "$RAMPART_HOME/bin/pip3r" ] \
    && [ -e "$RAMPART_HOME/modules/rampart-python.so" ] || modules+=(rampart-python)
if [ $PLATFORM = linux ]; then
    llama=$(readlink "$RAMPART_HOME/modules/rampart-llamacpp.so" 2>/dev/null || true)
    case $llama in
        *_cu13.so) ;;
        *_cu12.so) die "this rampart has llama.cpp's CUDA 12 build; Rampart Image Studio needs the CUDA 13 one:
           $RAMPART --install rampart-langtools-cu13
       (this switches every program that uses this rampart to the CUDA 13 build)" ;;
        *) modules+=(rampart-langtools-cu13) ;;   # missing, or the CPU build
    esac
else
    [ -e "$RAMPART_HOME/modules/rampart-llamacpp.so" ] || modules+=(rampart-langtools)
fi
if [ ${#modules[@]} -gt 0 ]; then
    if [ -w "$RAMPART_HOME/bin" ] && [ -w "$RAMPART_HOME/modules" ]; then
        say "Installing rampart modules: ${modules[*]}"
        "$RAMPART" --install "${modules[@]}" || die "rampart --install ${modules[*]} failed"
    else
        die "rampart at $RAMPART_HOME needs ${modules[*]}, and you cannot write to it.  Either run
           sudo $RAMPART --install ${modules[*]}
       or install a rampart of its own, in ./rampart:  ./setup.sh --own-rampart
       or use another rampart of your own:  RAMPART=/path/to/bin/rampart ./setup.sh"
    fi
fi

PY3R=$RAMPART_HOME/bin/python3r PIP3R=$RAMPART_HOME/bin/pip3r
pyver=$("$PY3R" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
[ "$pyver" = 3.11 ] || die "rampart's python is $pyver; Rampart Image Studio's packages are pinned for 3.11"
echo "Rampart Image Studio setup: $PLATFORM, rampart $(rampart_version "$RAMPART"), python $pyver"

# --- python packages ----------------------------------------------------------------
# Pinned to the versions Rampart Image Studio is tested with.  TORCH_INDEX picks the CUDA build
# (cu130 needs an NVIDIA driver from the 580 series or newer).
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
if [ $PLATFORM = linux ]; then
    REQS="torch==2.14.1 @$TORCH_INDEX; transformers==5.20.2 safetensors==0.8.0 pillow==13.0.0 numpy==2.4.6; comfy-kitchen==0.2.37 --no-deps"
else
    REQS="mflux==0.23.0 mlx==0.32.4 pillow<13"
fi
PYLIB=engine/pylib
STAMP=$PYLIB/.ris-requirements

if [ $PYTHON = 1 ]; then
    if [ -f $STAMP ] && [ "$(cat $STAMP)" = "$REQS" ]; then
        say "Python packages: up to date ($PYLIB)"
    else
        say "Python packages -> $PYLIB (a few GB; rampart's own site-packages are not touched)"
        rm -rf $PYLIB engine/pycache
        if [ $PLATFORM = linux ]; then
            "$PIP3R" install --target $PYLIB torch==2.14.1 --index-url "$TORCH_INDEX"
            "$PIP3R" install --target $PYLIB transformers==5.20.2 safetensors==0.8.0 pillow==13.0.0 numpy==2.4.6
            "$PIP3R" install --target $PYLIB --no-deps comfy-kitchen==0.2.37   # INT8 kernels
        else
            "$PIP3R" install --target $PYLIB mflux==0.23.0 mlx==0.32.4 "pillow<13"
        fi
        echo "$REQS" > $STAMP
    fi
    if [ $PLATFORM = linux ]; then
        # cuDNN loads its parts by their full version (libcudnn_graph.so.9.24.0) before the
        # major one, so a system cuDNN of the same version would be found first (Debian's
        # libcudnn9-cuda-12, built for CUDA 12); give PyTorch's copies those names as well
        cudnn=$(ls -d $PYLIB/nvidia_cudnn_cu13-*.dist-info 2>/dev/null | sed -E 's/.*-9\.([0-9]+\.[0-9]+).*/\1/' | head -1 || true)
        if [ -n "$cudnn" ]; then
            for f in $PYLIB/nvidia/cudnn/lib/libcudnn*.so.9; do ln -sfn "$(basename "$f")" "$f.$cudnn"; done
        fi
    fi
fi

# --- weights ------------------------------------------------------------------------
if command -v sha256sum >/dev/null; then sha256() { sha256sum "$1" | cut -d' ' -f1; }
else sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }; fi
fsize() { stat -c %s "$1" 2>/dev/null || stat -f %z "$1"; }

OK=weights/.verified   # "sha256 path" of files already checked
mkdir -p weights
touch $OK

verified() { grep -qx "$1 $2" $OK; }

fetch() {   # sha256 size dest sources...
    local sum=$1 size=$2 dest=weights/$3; shift 3
    if [ -f "$dest" ] && [ "$(fsize "$dest")" = "$size" ]; then
        if [ $VERIFY = 0 ] && verified "$sum" "$dest"; then return 0; fi
        printf '  checking  %s\n' "$dest"
        if [ "$(sha256 "$dest")" = "$sum" ]; then
            verified "$sum" "$dest" || echo "$sum $dest" >> $OK
            return 0
        fi
        echo "  checksum mismatch, downloading again: $dest"
        rm -f "$dest"
    fi
    mkdir -p "$(dirname "$dest")"
    local src repo path auth
    for src in "$@"; do
        repo=${src%%::*} path=${src#*::} auth=()
        if [ "${repo%%/*}" = black-forest-labs ]; then
            [ -n "${HF_TOKEN:-}" ] || continue
            auth=(-H "Authorization: Bearer $HF_TOKEN")
        fi
        printf '  fetching  %s  (%s MB from %s)\n' "$dest" $((size / 1048576)) "$repo"
        local progress=-sS
        [ "$size" -gt 104857600 ] && progress=--progress-bar
        # resume a partial download; a mirror holds the same bytes, so any of them can continue it
        if curl $progress -fL --retry 3 --retry-delay 5 -C - ${auth[@]+"${auth[@]}"} -o "$dest.part" \
                "https://huggingface.co/$repo/resolve/main/$path" \
            && [ "$(fsize "$dest.part")" = "$size" ] && [ "$(sha256 "$dest.part")" = "$sum" ]; then
            mv "$dest.part" "$dest"
            echo "$sum $dest" >> $OK
            return 0
        fi
        echo "  failed from $repo"
        # a complete but wrong file cannot be resumed; a short one can
        [ -f "$dest.part" ] && [ "$(fsize "$dest.part")" -ge "$size" ] && rm -f "$dest.part"
    done
    echo "setup: could not download $dest from any source" >&2
    [ -z "${HF_TOKEN:-}" ] && echo "  (set HF_TOKEN to also try black-forest-labs directly)" >&2
    return 1
}

if [ $WEIGHTS = 1 ]; then
    groups=" common $PLATFORM "
    [ $FP8 = 1 ] && groups+="fp8 "
    total=0
    while read -r group sum size dest rest; do
        [[ $group == \#* || $groups != *" $group "* ]] && continue
        [ -f "weights/$dest" ] && [ "$(fsize "weights/$dest")" = "$size" ] || total=$((total + size))
    done < weights.manifest
    say "Weights -> weights/ ($((total / 1073741824)) GB to download)"
    if [ $total -gt 0 ]; then
        free=$(df -Pk weights | awk 'NR == 2 {print $4 * 1024}')
        [ "$free" -gt $((total + 1073741824)) ] || die "not enough disk space in weights/ (need $((total / 1073741824)) GB)"
    fi
    failed=0
    while read -r group sum size dest rest; do
        [[ $group == \#* || $groups != *" $group "* ]] && continue
        # shellcheck disable=SC2086  # rest is the list of sources
        fetch "$sum" "$size" "$dest" $rest </dev/null || failed=1
    done < weights.manifest
    [ $failed = 0 ] || die "some weights are missing; run ./setup.sh again to retry"
    echo "  all weights present and verified"
fi

# --- check the modules load ---------------------------------------------------------
# with the CUDA libraries studio.sh gives the web server, not a CUDA toolkit's
CUDA_LIBS=$ROOT/engine/pylib/nvidia/cu13/lib
check_js=$(mktemp); TMPS+=("$check_js")
echo 'require("rampart-curl"); require("rampart-llamacpp");' > "$check_js"
if [ $PLATFORM = linux ] && [ ! -d "$CUDA_LIBS" ]; then
    echo "Not checking rampart-llamacpp: engine/pylib is not installed yet (./setup.sh without --no-python)"
elif ! LD_LIBRARY_PATH=$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} "$RAMPART" "$check_js" >/dev/null 2>&1; then
    die "rampart-llamacpp or rampart-curl does not load; try: $RAMPART $check_js"
fi

# --- local settings -----------------------------------------------------------------
# remember the rampart (studio.sh and engine/engine.sh use it and its python3r)
conf_set RAMPART "$RAMPART" "the rampart that Rampart Image Studio runs on; set by setup.sh"
mkdir -p web_server/data web_server/logs

say "Done."
# the engine's torch.compile needs a C compiler; without one it runs uncompiled (engine/engine.sh)
if [ $PLATFORM = linux ] && [ -z "${CC:-}" ] && ! command -v gcc >/dev/null && ! command -v clang >/dev/null; then
    echo "Note: no C compiler (gcc or clang) found.  Rampart Image Studio works without one, but the"
    echo "      image engine then runs about 10% slower.  For full speed install one:"
    echo "          sudo apt install gcc     (Debian, Ubuntu)"
    echo "          sudo dnf install gcc     (Fedora, RHEL)"
fi
echo "Start Rampart Image Studio with:  ./studio.sh start"
