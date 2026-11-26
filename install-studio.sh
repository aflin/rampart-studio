#!/bin/sh
# Rampart Image Studio installer
#
#   curl -fsSL https://rampart.dev/install-studio.sh | sh
#   curl -fsSL https://rampart.dev/install-studio.sh | sh -s -- [DIR] [setup.sh options]
#
# Downloads the latest release from GitHub into DIR (asked for, the default being
# ./rampart-studio, when DIR is not given and there is a terminal), then runs
# its setup.sh, which installs Rampart if needed, the Python packages and the model weights
# (about 26 GB), and asks for the web address and GPUs.  Run it again on an install to
# update that install to the latest release: its settings, packages, weights and images
# are kept.
#
# Environment:  STUDIO_VERSION=v0.1.0  install that release instead of the latest
#
# POSIX sh, for every shell "curl | sh" may land in.
set -eu

REPO=aflin/rampart-studio
die() { printf 'install-studio: %s\n' "$*" >&2; exit 1; }
for tool in curl tar; do
    command -v $tool >/dev/null 2>&1 || die "$tool is needed, and it is not installed"
done

dir=
case ${1:-} in ""|-*) ;; *) dir=$1; shift ;; esac
have_tty() { (: < /dev/tty) 2>/dev/null; }

# the release: the tag GitHub's "latest release" page redirects to
version=${STUDIO_VERSION:-}
if [ -z "$version" ]; then
    url=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest") \
        || die "could not reach GitHub"
    case $url in
        */releases/tag/*) version=${url##*/} ;;
        *) die "no release found at https://github.com/$REPO/releases" ;;
    esac
fi

# DIR must be new (or empty), or an earlier install of a release, which is updated.
# Sets problem (why DIR cannot be used, or empty) and update (the release DIR has).
check_dir() {
    problem= update=
    [ -e "$1" ] || return 0
    if [ -d "$1/.git" ]; then
        problem="$1 is a git checkout; update it with  git pull  and run its ./setup.sh"
    elif [ -f "$1/setup.sh" ] && [ -f "$1/studio.sh" ]; then
        update=$(cat "$1/.studio-release" 2>/dev/null || echo "an earlier release")
    elif [ -n "$(ls -A "$1" 2>/dev/null)" ]; then
        problem="$1 exists and is not a Rampart Image Studio install; choose another directory"
    fi
}

if [ -z "$dir" ]; then
    dir=$PWD/rampart-studio
    if have_tty; then
        while :; do
            printf 'Install Rampart Image Studio %s into [%s]: ' "$version" "$dir" > /dev/tty
            read -r answer < /dev/tty || answer=
            answer=${answer:-$dir}
            case $answer in "~"|"~/"*) answer=$HOME${answer#"~"} ;; esac
            check_dir "$answer"
            [ -z "$problem" ] && { dir=$answer; break; }
            echo "  $problem" > /dev/tty
        done
    fi
fi
check_dir "$dir"
[ -z "$problem" ] || die "$problem"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if [ "$update" = "$version" ]; then
    echo "$dir already has Rampart Image Studio $version; reinstalling its files (settings, packages, weights and images are kept)"
elif [ -n "$update" ]; then
    echo "Updating $dir from $update to Rampart Image Studio $version (settings, packages, weights and images are kept)"
else
    echo "Installing Rampart Image Studio $version into $dir"
fi
curl -fsSL "https://github.com/$REPO/archive/refs/tags/$version.tar.gz" -o "$tmp/release.tar.gz" \
    || die "could not download release $version"
mkdir "$tmp/x"
tar xzf "$tmp/release.tar.gz" -C "$tmp/x" || die "could not unpack release $version"
src=$(find "$tmp/x" -mindepth 1 -maxdepth 1 -type d | head -1)
[ -n "$src" ] && [ -f "$src/setup.sh" ] || die "release $version does not look like Rampart Image Studio"
mkdir -p "$dir"
cp -R "$src/." "$dir/"
echo "$version" > "$dir/.studio-release"
dir=$(cd "$dir" && pwd)

# setup.sh asks its questions on the terminal: when piped from curl, stdin is the pipe
cd "$dir"
rc=0
if have_tty; then
    ./setup.sh "$@" < /dev/tty || rc=$?
else
    ./setup.sh "$@" || rc=$?
fi
echo
if [ $rc -ne 0 ]; then
    echo "setup did not finish; fix the problem above and run it again:  cd $dir && ./setup.sh"
elif [ -n "$update" ]; then
    echo "Updated $dir to $version.  If Rampart Image Studio is running, restart it:"
    echo "    cd $dir && ./studio.sh restart"
else
    echo "Installed in $dir.  Start it with:"
    echo "    cd $dir && ./studio.sh start"
fi
exit $rc
