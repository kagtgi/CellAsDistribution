#!/usr/bin/env bash
# kill processes whose command line matches $1, excluding this script and its parent shell
for p in $(pgrep -f "$1"); do
  [ "$p" = "$$" ] || [ "$p" = "$PPID" ] && continue
  kill "$p" 2>/dev/null && echo "killed $p"
done
exit 0
