#!/bin/bash
#SBATCH --job-name=test-cluster-no-container
#SBATCH --output=/home/testuser/.sky_provision/slurm-%j.out
#SBATCH --error=/home/testuser/.sky_provision/slurm-%j.out
#SBATCH --nodes=1
#SBATCH --wait-all-nodes=1
# Let the job be terminated rather than requeued implicitly.
#SBATCH --no-requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=8192M

#SBATCH --time=7-00:00:00

# Cleanup function to remove cluster dirs on job termination.
cleanup() {
    saved_exit=$?
    echo "Terminating Skylet service..."
    if [ -n "${SKYLET_SERVICE_PID:-}" ]; then
        kill "$SKYLET_SERVICE_PID" 2>/dev/null || true
    fi
    if [ -s "/tmp/test-cluster-no-container/.sky/skylet_pid" ]; then
        skylet_pid=$(cat "/tmp/test-cluster-no-container/.sky/skylet_pid")
        if [ "$skylet_pid" -gt 0 ] 2>/dev/null &&                 ps -p "$skylet_pid" -o args= | grep -q "sky.skylet.skylet"; then
            kill "$skylet_pid" 2>/dev/null || true
        fi
    fi
    if [ -n "${SKYLET_SERVICE_PID:-}" ]; then
        wait "$SKYLET_SERVICE_PID" 2>/dev/null || true
    fi
    if [ -n "${CONTAINER_PID:-}" ]; then
        kill "$CONTAINER_PID" 2>/dev/null || true
        wait "$CONTAINER_PID" 2>/dev/null || true
    fi
    echo "Cleaning up sky directories..."
    # Remove the per-node enroot container, if it exists.
    # This is only needed when container_scope=global.
    # When container_scope=job, named containers are removed automatically
    # at the end of the Slurm job, see: https://github.com/NVIDIA/pyxis/wiki/Setup#slurm-epilog
    srun --nodes=1 --ntasks-per-node=1 enroot remove -f pyxis_test-cluster-no-container 2>/dev/null || true
    # Clean up sky runtime directory on each node.
    # NOTE: We can do this because --nodes for both this srun and the
    # sbatch is the same number. Otherwise, there are no guarantees
    # that this srun will run on the same subset of nodes as the srun
    # that created the sky directories.
    srun --nodes=1 --ntasks-per-node=1 rm -rf -- /tmp/test-cluster-no-container
    rm -rf /home/testuser/.sky_clusters/test-cluster-no-container
    exit $saved_exit
}
# Run cleanup on any exit, including container init failures.
trap cleanup EXIT
# On SIGTERM (job cancellation via scancel), exit 0 so cleanup treats
# it as a graceful shutdown rather than propagating an error code.
trap 'exit 0' TERM

# Create sky home directory and subdirectories for the cluster.
mkdir -p /home/testuser/.sky_clusters/test-cluster-no-container/sky_logs /home/testuser/.sky_clusters/test-cluster-no-container/sky_workdir /home/testuser/.sky_clusters/test-cluster-no-container/.sky
# A new allocation must not inherit jobs, PIDs, or version files from a prior
# allocation whose EXIT trap could not run (for example after SIGKILL).
srun --nodes=1 --ntasks-per-node=1 rm -rf -- /tmp/test-cluster-no-container
srun --nodes=1 --ntasks-per-node=1 mkdir -p -- /tmp/test-cluster-no-container
# Marker file to indicate we're in a Slurm cluster.
touch /home/testuser/.sky_clusters/test-cluster-no-container/.sky_slurm_cluster
# Store proctrack type for task executor to read.
echo 'cgroup' > /home/testuser/.sky_clusters/test-cluster-no-container/.sky_proctrack_type
# Suppress login messages.
touch /home/testuser/.sky_clusters/test-cluster-no-container/.hushlogin
rm -f /home/testuser/.sky_clusters/test-cluster-no-container/.sky/skylet_start.sh /tmp/test-cluster-no-container/.sky/skylet_service
SKYLET_START=$SECONDS
echo "[skylet] Starting service on $SLURMD_NODENAME"
bash -c 'export SKY_RUNTIME_DIR=/tmp/test-cluster-no-container
cd /home/testuser/.sky_clusters/test-cluster-no-container
export HOME="$PWD"
mkdir -p /tmp/test-cluster-no-container/.sky
touch /tmp/test-cluster-no-container/.sky/skylet_service
while [ ! -s /home/testuser/.sky_clusters/test-cluster-no-container/.sky/skylet_start.sh ]; do sleep 1; done
while true; do
    bash /home/testuser/.sky_clusters/test-cluster-no-container/.sky/skylet_start.sh
    sleep 1
done
' &
SKYLET_SERVICE_PID=$!
while [ ! -f /tmp/test-cluster-no-container/.sky/skylet_service ]; do
  if ! kill -0 "$SKYLET_SERVICE_PID" 2>/dev/null; then
    echo "[skylet] ERROR: Service failed to start."
    wait "$SKYLET_SERVICE_PID"
    exit $?
  fi
  sleep 1
done
echo "[skylet] Service ready in $((SECONDS - SKYLET_START))s"

touch /home/testuser/.sky_clusters/test-cluster-no-container/.sky_sbatch_ready_$SLURM_JOB_ID
wait "$SKYLET_SERVICE_PID"
