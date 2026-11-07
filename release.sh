#!/usr/bin/env bash
# Publish a release of Rampart Image Studio on GitHub: pushes main, tags it and creates the
# release (with GitHub's tar.gz and zip of the source).  Run from a clean checkout of main.
#   ./release.sh            asks for the version (Enter takes the suggested next one)
#   ./release.sh 0.2.0      that version (still asks before publishing)
set -euo pipefail
cd "$(dirname "$0")"
die() { echo "release: $*" >&2; exit 1; }

command -v gh >/dev/null || die "gh (GitHub CLI) is not installed"
gh auth status >/dev/null 2>&1 || die "gh is not logged in: run  gh auth login"
[ "$(git branch --show-current)" = main ] || die "not on main"
[ -z "$(git status --porcelain)" ] || die "uncommitted changes; commit them first:
$(git status --short)"
git fetch -q origin
[ "$(git rev-list --count HEAD..origin/main)" = 0 ] || die "origin/main has commits this checkout does not; git pull first"

# the next version: the newest vX.Y.Z tag with its last number raised, or v0.1.0
last=$(git tag -l 'v[0-9]*.[0-9]*.[0-9]*' --sort=-v:refname | head -1)
if [ -n "$last" ]; then
    IFS=. read -r major minor patch <<< "${last#v}"
    next=v$major.$minor.$((patch + 1))
else
    next=v0.1.0
fi

version=${1:-}
if [ -z "$version" ]; then
    echo "Latest release: ${last:-none}"
    read -r -p "Version [$next]: " version
    version=${version:-$next}
fi
version=v${version#v}
[[ $version =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "$version: give a version like 1.2.3"
git rev-parse -q --verify "refs/tags/$version" >/dev/null && die "$version already exists"

echo "Release $version from $(git log -1 --format='%h %s')"
unpushed=$(git rev-list --count origin/main..HEAD)
[ "$unpushed" = 0 ] || echo "  (also pushes $unpushed commit(s) to origin/main)"
read -r -p "Publish? [y/N] " ok
[[ $ok =~ ^[Yy] ]] || { echo "Nothing done."; exit 0; }

git push -q origin main
git tag -a "$version" -m "Rampart Image Studio ${version#v}"
git push -q origin "$version"
gh release create "$version" --verify-tag --title "Rampart Image Studio ${version#v}" --notes ""
echo "Download: https://github.com/aflin/rampart-studio/archive/refs/tags/$version.tar.gz"
