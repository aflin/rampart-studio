#!/usr/bin/env bash
# Start/stop/status for RIS Engine.  Extra args go to server.py, e.g.:
#   ./engine.sh start                       # listens on 127.0.0.1:8189
#   ./engine.sh start --port 8190 --dit weights/flux-3-image-klein/variants/fp8r/model.safetensors
#   ./engine.sh restart                     # restarts with the args it was started with
cd "$(dirname "$(readlink -f "$0")")" || exit 1
# compiled bytecode goes here, never next to rampart's python (which may be writable)
export PYTHONPYCACHEPREFIX=$PWD/pycache
# python3r is a wrapper script that runs rampart's python as a child; launch the
# interpreter itself so the pid we record is the server's.  PYTHON3R (set by studio.sh)
# is the python3r of the rampart that Rampart Image Studio uses; otherwise the one on the PATH.
PY3R=${PYTHON3R:-$(command -v python3r || echo /usr/local/rampart/bin/python3r)}
PY=${PYTHON:-$("$PY3R" -c 'import sys; print(sys.executable)')}
# the CUDA and cuDNN libraries from PyTorch's packages, first: cuDNN and some kernels load
# libraries by name, which would otherwise find a CUDA toolkit's copies (possibly built
# for another CUDA version) through the system's library cache
for d in "$PWD"/pylib/nvidia/*/lib; do
    [ -d "$d" ] && LD_LIBRARY_PATH=$d${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
done
export LD_LIBRARY_PATH
PIDFILE=engine.pid ARGSFILE=engine.args LOG=engine.log   # previous run's log kept as .old
cmd=$1; shift

# torch.compile (Linux) builds a small C launcher for its GPU kernels the first time, which needs
# a C compiler: $CC, gcc or clang, as Triton looks for them
have_cc() {
    if [ -n "${CC:-}" ]; then command -v "$CC" >/dev/null; else command -v gcc >/dev/null || command -v clang >/dev/null; fi
}

running() { [ -f $PIDFILE ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; }

usage() {
    cat <<EOF
usage: $0 {start|stop|restart|status|help} [server.py options]

  start [opts]    start the engine in the background and wait until it listens
  stop            stop it (forced after ~10s)
  restart [opts]  stop and start; with no opts, reuses the options from the last start
  status          show pid, options, load and listening address (exit 3 if not running)
  help            this message; 'help server' lists every server.py option

Files (in $PWD): engine.pid, engine.args, engine.log (previous run: .log.old)
EOF
}

start() {
    if running; then echo "RIS Engine already running (pid $(cat $PIDFILE))"; return 1; fi
    if [ $# -gt 0 ]; then printf '%q ' "$@" > $ARGSFILE; else : > $ARGSFILE; fi
    # without a C compiler torch.compile cannot start, so run uncompiled (not saved in
    # engine.args: a restart after installing one compiles again)
    local extra=()
    if [ "$(uname -s)" = Linux ] && ! have_cc && [[ " $* " != *" --no-compile "* ]]; then
        echo "No C compiler (gcc or clang) found: the engine runs without torch.compile, about 10% slower."
        echo "  For full speed install one (sudo apt install gcc, or sudo dnf install gcc) and restart."
        extra=(--no-compile)
    fi
    [ -f $LOG ] && mv $LOG $LOG.old
    : > $LOG   # now, not when the background job gets to its redirection, for the checks below
    # its own session where setsid exists (Linux), so it outlives the terminal's process group
    $(command -v setsid) nohup "$PY" -u server.py "$@" ${extra[@]+"${extra[@]}"} >> $LOG 2>&1 &
    echo $! > $PIDFILE
    for _ in $(seq 300); do
        if ! running; then
            # the last line of a Python traceback is the error itself
            echo "RIS Engine failed to start: $(grep -v '^[[:space:]]*$' $LOG | tail -1)"
            echo "  (see engine/$LOG)"
            rm -f $PIDFILE; return 1
        fi
        if grep -q "RIS Engine listening on" $LOG; then
            echo "RIS Engine started (pid $(cat $PIDFILE)): $(grep -o 'http://.*' $LOG | tail -1)"
            return 0
        fi
        sleep 1
    done
    echo "RIS Engine is still starting after 5 minutes (pid $(cat $PIDFILE)); see $LOG"
    return 1   # not ready yet (it goes on loading)
}

stop() {
    if ! running; then echo "RIS Engine not running"; rm -f $PIDFILE; return 0; fi
    local pid; pid=$(cat $PIDFILE)
    kill "$pid"
    for _ in $(seq 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
    rm -f $PIDFILE
    echo "RIS Engine stopped"
}

# address:port the engine listens on (ss on Linux, lsof on macOS)
listening() {
    local pid; pid=$(cat $PIDFILE)
    if command -v ss >/dev/null; then
        ss -ltnp 2>/dev/null | grep "pid=$pid," | awk '{print $4}'
    else
        lsof -nP -a -p "$pid" -iTCP -sTCP:LISTEN 2>/dev/null | awk 'NR > 1 {print $9}'
    fi
}

status() {
    if running; then
        echo "RIS Engine running (pid $(cat $PIDFILE)): $(cat $ARGSFILE 2>/dev/null)"
        local addr; addr=$(listening | head -1)
        [ -n "$addr" ] && echo "  listening on $addr" && curl -s --max-time 2 "http://${addr/0.0.0.0/127.0.0.1}/info" && echo
    else
        echo "RIS Engine not running"; return 3
    fi
}

case $cmd in
    start)   start "$@" ;;
    stop)    stop ;;
    restart) [ $# -eq 0 ] && [ -f $ARGSFILE ] && eval "set -- $(cat $ARGSFILE)"; stop; start "$@" ;;
    status)  status ;;
    help|-h|--help) if [ "$1" = server ]; then "$PY" server.py --help; else usage; fi ;;
    *)       usage >&2; exit 1 ;;
esac
