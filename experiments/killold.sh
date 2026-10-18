#!/usr/bin/env bash
# stop the first finish script (it would power the VM off before the extra queue ends); never matches this script's own command line
for p in $(pgrep -f "finish_and_stop.sh"); do
  [ "$p" != "$$" ] && kill "$p" 2>/dev/null
done
sleep 1
echo "remaining: $(pgrep -f finish_and_stop.sh | grep -v $$ | wc -l)"
exit 0
