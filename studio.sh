#!/usr/bin/env bash
# Start/stop/status for Rampart Image Studio: the image engine (engine/) and the Rampart web
# server with the prompt enhancer (web_server/).  Settings: studio.conf.
#   ./studio.sh {start|stop|restart|status}
cd "$(dirname "$0")" || exit 1
ROOT=$PWD

RAMPART= WEB_PORT=8088 HOST= BIND_ALL= ENGINE_PORT=8189 ENGINE_GPU= ENHANCER_GPU= ENHANCER_SEQS=8 ENHANCER_GPU_LAYERS=all ENGINE_ARGS=
[ -f studio.conf ] && . ./studio.conf
# HOST: the address to listen on, 0.0.0.0 for every interface (settings from before HOST: BIND_ALL=1)
[ -n "$HOST" ] || { [ "$BIND_ALL" = 1 ] && HOST=0.0.0.0 || HOST=127.0.0.1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # GPU numbers match nvidia-smi
export RIS_WEB_PORT=$WEB_PORT RIS_HOST=$HOST RIS_ENGINE_PORT=$ENGINE_PORT RIS_ENHANCER_SEQS=$ENHANCER_SEQS
export RIS_ENHANCER_GPU_LAYERS=$ENHANCER_GPU_LAYERS

# the rampart setup.sh chose (studio.conf), and the python3r of the same install
[ -n "$RAMPART" ] || RAMPART=$(command -v rampart || echo /usr/local/rampart/bin/rampart)
resolve() {
    local p=$1 t
    while [ -L "$p" ]; do
        t=$(readlink "$p")
        case $t in /*) p=$t ;; *) p=$(dirname "$p")/$t ;; esac
    done
    echo "$(cd "$(dirname "$p")" && pwd -P)/$(basename "$p")"
}
export PYTHON3R
PYTHON3R=$(dirname "$(resolve "$RAMPART")")/python3r
# the prompt enhancer's llama.cpp (CUDA 13 build) runs on the CUDA libraries PyTorch
# installed into engine/pylib, so no CUDA toolkit is needed
CUDA_LIBS=$ROOT/engine/pylib/nvidia/cu13/lib
# where to reach the web server from here, and the address to show for it
if [ "$HOST" = 0.0.0.0 ]; then CHECK_HOST=127.0.0.1 SHOW_HOST=$(hostname); else CHECK_HOST=$HOST SHOW_HOST=$HOST; fi

web() {
    mkdir -p web_server/data web_server/logs
    [ -d "$CUDA_LIBS" ] && local -x LD_LIBRARY_PATH=$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    if [ "$(uname -s)" = Darwin ]; then
        web_foreground "$@"
    elif [ -n "$ENHANCER_GPU" ]; then
        CUDA_VISIBLE_DEVICES=$ENHANCER_GPU "$RAMPART" web_server/web_server_conf.js "$@"
    else
        "$RAMPART" web_server/web_server_conf.js "$@"
    fi
}

# macOS: rampart's daemon mode starts the server through launchd, which an ssh session cannot
# use (and a forked daemon loses Metal), so run it in the foreground under nohup instead
web_foreground() {
    local pidfile=web_server/server.fg.pid pid
    pid=$(cat $pidfile 2>/dev/null)
    alive() { [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; }
    case $1 in
        start)
            if alive; then echo "Server is already running."; return; fi
            RIS_FOREGROUND=1 nohup "$RAMPART" web_server/web_server_conf.js start >> web_server/logs/server.out 2>&1 &
            pid=$!
            echo $pid > $pidfile
            # a failure to start (a port in use, a bad setting) ends it within a moment
            for _ in 1 2 3 4 5 6; do sleep 0.5; alive || break; done
            if ! alive; then
                rm -f $pidfile
                echo "The web server failed to start: $(grep -E 'Error|error' web_server/logs/server.out | tail -1) (see web_server/logs/server.out)"
                return 1
            fi
            echo "Server has been started." ;;
        stop)
            if alive; then
                kill "$pid"
                for _ in $(seq 20); do alive || break; sleep 0.5; done
            fi
            rm -f $pidfile
            echo "Server has been stopped." ;;
        status)
            if alive; then echo "server is running"; else echo "server is not running"; fi ;;
    esac
}

engine() {
    if [ -n "$ENGINE_GPU" ]; then
        CUDA_VISIBLE_DEVICES=$ENGINE_GPU engine/engine.sh "$@"
    else
        engine/engine.sh "$@"
    fi
}

# who listens on port $1: "pid N: command", "another program" when that is unknown, or nothing
port_owner() {
    local pid=
    command -v lsof >/dev/null && pid=$(lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2>/dev/null | head -1)
    [ -z "$pid" ] && command -v ss >/dev/null \
        && pid=$(ss -ltnpH "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
    if [ -n "$pid" ]; then
        echo "pid $pid: $(ps -o command= -p "$pid" 2>/dev/null | cut -c1-160)"
    elif (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null || { [ "$HOST" != 0.0.0.0 ] && (exec 3<>"/dev/tcp/$HOST/$1") 2>/dev/null; }; then
        echo "another program (another user's, or not visible to lsof/ss)"
    fi
}

start() {
    # curl tells when the web server is up
    command -v curl >/dev/null || { echo "studio.sh needs curl to start Rampart Image Studio; install it (e.g. sudo apt install curl)"; return 1; }
    # what of this install is already running is left running
    local web_up=0 engine_up=0
    web status 2>/dev/null | grep -q "is running" && web_up=1
    engine status >/dev/null 2>&1 && engine_up=1
    if [ $web_up = 1 ] && [ $engine_up = 1 ]; then
        echo "Rampart Image Studio is already running: http://$SHOW_HOST:$WEB_PORT/studio/  (./studio.sh restart to restart it)"
        return 0
    fi
    # a port held by something else (another program, or a Rampart Image Studio started from another
    # directory) would make the new server fail, while startup took the old one's answers for its own
    local port owner f ours=" "
    for f in web_server/server.pid web_server/server.fg.pid engine/engine.pid; do   # (rampart's has no newline)
        [ -f $f ] && ours+="$(cat $f) "
    done
    for port in "$WEB_PORT" "$ENGINE_PORT"; do
        owner=$(port_owner "$port")
        [ -n "$owner" ] || continue
        case $owner in "pid "*) [[ $ours == *" $(echo "$owner" | cut -d' ' -f2 | tr -d :) "* ]] && continue ;; esac
        echo "Port $port is already in use by $owner"
        echo "Stop that first, or choose other ports (./setup.sh for the web port; ENGINE_PORT in studio.conf)."
        return 1
    done
    # the web server first: the engine sizes itself to the GPU memory the prompt enhancer leaves
    if [ $web_up = 1 ]; then
        echo "Web server already running."
    else
        web start | tail -1
        [ "${PIPESTATUS[0]}" = 0 ] || return 1
    fi
    # it answers once the enhancer model has loaded (preThreadFunc)
    echo "Loading the prompt enhancer..."
    local up=0
    for _ in $(seq 120); do
        curl -s -o /dev/null --max-time 2 "http://$CHECK_HOST:$WEB_PORT/studio/" && { up=1; break; }
        sleep 1
    done
    [ $up = 1 ] || { echo "The web server did not answer within 2 minutes; see web_server/logs/error.log"; return 1; }
    if [ $engine_up = 1 ]; then
        echo "RIS Engine already running."
    else
        echo "Starting the image engine..."
        # shellcheck disable=SC2086  # ENGINE_ARGS is a list of options
        engine start --port "$ENGINE_PORT" --out "$ROOT/web_server/data/ris/images" $ENGINE_ARGS || return 1
    fi
    echo "Rampart Image Studio ready: http://$SHOW_HOST:$WEB_PORT/studio/"
}

stop() {
    web stop | tail -1
    engine stop
}

status() {
    local rc=0
    engine status || rc=3
    if web status | grep -q "is running"; then
        echo "Web server running: http://$SHOW_HOST:$WEB_PORT/studio/"
    else
        echo "Web server not running"; rc=3
    fi
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.used,memory.total \
        --format=csv,noheader 2>/dev/null | sed "s/^/  GPU /"
    return $rc
}

case $1 in
    start)   start ;;
    stop)    stop ;;
    restart) stop; start ;;
    status)  status ;;
    *)       echo "usage: $0 {start|stop|restart|status}"; exit 1 ;;
esac
