#!/bin/bash

FOLDER1="all_in_one_results_bgm_cut"
FOLDER2="RapBank/data/vocal_cut"

start_time=$(date +%s)

# Get total once at start (assumed fixed)
total=$(ls -1 "$FOLDER2" 2>/dev/null | wc -l)

while true; do
    now=$(date +%s)
    current=$(ls -1 "$FOLDER1" 2>/dev/null | wc -l)

    # Compute percent
    if [ "$total" -eq 0 ]; then
        percent=0
    else
        percent=$(( 100 * current / total ))
    fi

    # Compute ETA
    elapsed=$(( now - start_time ))

    if [ "$current" -gt 0 ]; then
        rate=$(echo "$elapsed / $current" | bc -l)  # seconds per file
        remaining=$(( total - current ))
        eta_seconds=$(printf "%.0f" "$(echo "$rate * $remaining" | bc -l)")
    else
        eta_seconds=0
    fi

    # Format ETA as H:MM:SS
    eta_formatted=$(printf "%02d:%02d:%02d" \
        $((eta_seconds/3600)) \
        $(( (eta_seconds%3600)/60 )) \
        $((eta_seconds%60)) )

    # Build progress bar (50 chars)
    bar_width=50
    filled=$(( bar_width * percent / 100 ))
    empty=$(( bar_width - filled ))
    bar="$(printf '%*s' "$filled" '' | tr ' ' '#')"
    bar="$bar$(printf '%*s' "$empty" '' | tr ' ' '.')"

    clear
    echo "Progress: $current / $total  ($percent%)"
    echo "[$bar]"
    echo "ETA: $eta_formatted"
    echo "Elapsed: $elapsed sec"
    echo "Updated: $(date)"
    sleep 10
done
