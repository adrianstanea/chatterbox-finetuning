#!/bin/bash
# Auto-restart training loop for Chatterbox Romanian fine-tuning
# Automatically resumes from latest checkpoint after crashes

restart_count=0
MAX_CONSECUTIVE_FAST_FAILURES=5
FAST_FAILURE_THRESHOLD=120  # seconds

echo "Starting auto-restart training loop"
echo "Press Ctrl+C to stop"
echo "============================================"

while true; do
    start_time=$(date +%s)

    echo ""
    echo "========================================="
    echo "Training run #$((restart_count + 1))"
    echo "Started at: $(date)"
    echo "========================================="


    # Delete problematic files from the latest checkpoint if they exist (PyTorch 2.6 workaround)
    latest_ckpt_dir=$(ls -d chatterbox_output/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
    if [ -n "$latest_ckpt_dir" ]; then
        for f in "$latest_ckpt_dir"/rng_state*.pth "$latest_ckpt_dir"/optimizer.pt "$latest_ckpt_dir"/scheduler.pt; do
            if [ -f "$f" ]; then
                echo "Deleting $f to avoid PyTorch 2.6 checkpoint loading issue."
                rm -f "$f"
            fi
        done
    fi

    torchrun --nproc_per_node=4 train.py 2>&1 | tee -a training.log
    exit_code=${PIPESTATUS[0]}
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo ""
    echo "========================================="
    echo "Training exited with code $exit_code"
    echo "Duration: ${duration}s"
    echo "Ended at: $(date)"
    echo "========================================="

    # Check for signal-based exits
    if [ $exit_code -ge 128 ]; then
        signal=$((exit_code - 128))
        echo "WARNING: Training process was killed by signal $signal."
        case $signal in
            9)
                echo "Likely killed by SIGKILL (OOM or manual kill)."
                ;;
            11)
                echo "Likely segmentation fault (SIGSEGV)."
                ;;
            *)
                echo "See 'kill -l' for signal meanings."
                ;;
        esac
    fi

    # Check for successful completion
    if [ $exit_code -eq 0 ]; then
        echo "Training completed successfully!"
        break
    fi

    # Check for fast failures (might indicate configuration issue)
    if [ $duration -lt $FAST_FAILURE_THRESHOLD ]; then
        echo "WARNING: Fast failure detected (< ${FAST_FAILURE_THRESHOLD}s)"
    fi

    restart_count=$((restart_count + 1))

    # Wait before restart
    wait_time=10
    echo "Waiting ${wait_time}s before restart #$((restart_count + 1))..."
    sleep $wait_time
done

echo ""
echo "Training loop ended. Total restarts: $restart_count"
