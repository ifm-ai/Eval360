#!/usr/bin/env bash
# Bring up a single-node Slurm cluster INSIDE A CONTAINER.
#
# WHAT THIS IS FOR
# ----------------
# `scheduler/slurm_manager.py` shells out to sbatch/squeue/scancel/sinfo/sacct
# and parses their output. Nothing has ever run any of those binaries in CI:
# `tests/test_job_manager.py` patches `asyncio.create_subprocess_exec` and hands
# the parsers strings a test author typed, and `tests/fake_slurm.py` replaces the
# class outright. So every parser is only ever checked against fixtures written
# by whoever wrote the parser.
#
# This is the Slurm analogue of `kind create cluster`, and the analogy is exact
# in the part that matters: kind does not apt-install a kubelet on your machine,
# it runs Kubernetes in a container so the host stays clean. This script does
# the same. It is NOT run on the host — not on a GitHub runner, not on your
# laptop, not on a self-hosted runner. `scripts/slurm_cluster.sh` runs it inside
# a throwaway container, and that is the only supported way to invoke it.
#
# WHY NOT JUST RUN IT ON THE RUNNER. An earlier draft did, with sudo, and its
# entire cleanup story was "the runner is disposable". That is a bad bargain
# even when it holds: the script mutates /etc/slurm, /etc/munge and
# /var/lib/mysql and leaves root daemons running, so it cannot be torn down,
# cannot be run twice on one machine, breaks on a self-hosted runner, and can
# never be run on a workstation — which also means the local and CI clusters
# would have had to be configured separately, and would have drifted. In a
# container the cluster is disposable BY CONSTRUCTION rather than by luck, the
# same image runs in both places, and teardown is `docker rm -f`.
#
# Consequently this script assumes it is root in a container it is allowed to
# wreck, and refuses to run otherwise.
#
# WHAT THIS CLUSTER REQUIRES, AND WHY. These began as open questions during the
# spike that built this; all three are now settled by measurement, and each is
# load-bearing rather than a preference.
#
#   No systemd.  Every daemon is started directly instead of through the
#                packaged units, because a container has no systemd to delegate
#                to. slurm-wlm 23.11.4 is content with that, but note the
#                cgroup consequence in cgroup.conf below.
#
#   Fake GPU     A count-only GRES does NOT work. `Name=gpu Count=N` with no
#   DEVICE       `File=` is silently discarded by slurmd and the node registers
#   FILES.       INVALID_REG — see the gres.conf section. So the bring-up
#                mknods devices that nothing ever opens. This is unavoidable
#                rather than fastidious: `_serving_resource_args` ALWAYS emits
#                `--gres=gpu:N` and `gpus_per_node` is a StrictPositiveInt, so
#                no configuration can request zero, and neither a CI runner nor
#                a laptop has a GPU.
#
#   slurmdbd +   Not optional. `wait_for_terminal_job_outcomes` raises when
#   MariaDB.     sacct returns no root row, so a cluster without accounting
#                cannot exercise terminal-result code at all.

set -euo pipefail

# Four, not one: `update_allocation` allocates REPLICAS, and a single-GPU
# node cannot tell "submitted two replicas" from "submitted one" — the
# second would simply sit PENDING and every multi-replica assertion would
# pass for the wrong reason. Four leaves room for a scale-up test plus a
# deliberately-unschedulable job alongside it.
FAKE_GPUS="${FAKE_GPUS:-4}"
CONF_DIR="${CONF_DIR:-/etc/slurm}"
LOG_DIR="${SLURM_LOG_DIR:-/var/log/slurm}"
CLUSTER_NAME="${CLUSTER_NAME:-eval360ci}"
DB_PASSWORD="${DB_PASSWORD:-slurm}"

HOST="$(hostname -s)"

# CAPACITY IS DECLARED, NOT MEASURED — for the same reason the GPUs are.
#
# These were derived from `nproc` and /proc/meminfo, and that made the test
# cluster a different size on every machine. A GitHub runner reports CPUs=2
# where a developer laptop reports many more, so a test allocating three 1-CPU replicas
# passed locally and hung forever in CI waiting for a third job that could never
# start. That is exactly the local-versus-CI divergence this cluster exists to
# eliminate, reintroduced through the back door.
#
# `SlurmdParameters=config_overrides` (set below, already required for the fake
# GPUs) tells slurmd to believe the configuration rather than its own inventory,
# so the node can advertise capacity the host does not have. Nothing enforces
# it: cgroups are disabled here, so `--mem` and `--cpus-per-task` are accounting
# only, and the real load is a handful of tiny stub HTTP servers.
#
# Raise these if the suite ever needs more concurrency. Do NOT make them
# host-dependent again.
FAKE_CPUS="${FAKE_CPUS:-16}"
FAKE_MEM_MB="${FAKE_MEM_MB:-32768}"

_say() { printf '\n\033[1;36m=== %s\033[0m\n' "$*"; }

# --------------------------------------------------------------------------
# Refuse to wreck a machine that is not disposable.
#
# The container check is deliberately advisory-but-loud rather than silent: the
# failure this prevents is someone running the script on a workstation or a
# self-hosted runner and leaving munge keys, a Slurm config and a MariaDB
# instance behind. EVAL360_SLURM_ALLOW_HOST=1 is the escape hatch for a machine
# the caller genuinely intends to sacrifice.
# --------------------------------------------------------------------------
if [ "$(id -u)" -ne 0 ]; then
    echo "FAIL: this script must run as root inside a disposable container." >&2
    echo "      Use: scripts/slurm_cluster.sh up" >&2
    exit 1
fi
if [ ! -f /.dockerenv ] && [ ! -f /run/.containerenv ] \
   && ! grep -qE '(docker|containerd|kubepods|lxc)' /proc/1/cgroup 2>/dev/null; then
    if [ "${EVAL360_SLURM_ALLOW_HOST:-0}" != "1" ]; then
        echo "FAIL: this does not look like a container, and this script writes to" >&2
        echo "      /etc/slurm, /etc/munge and /var/lib/mysql and starts root daemons." >&2
        echo "      Use 'scripts/slurm_cluster.sh up', or set" >&2
        echo "      EVAL360_SLURM_ALLOW_HOST=1 if this machine is disposable." >&2
        exit 1
    fi
    echo "WARNING: running outside a container at the caller's request." >&2
fi

dump_logs() {
    printf '\n\033[1;31m=== slurm_up.sh FAILED — daemon logs follow\033[0m\n'
    for f in "$LOG_DIR"/*.log /var/log/munge/munged.log; do
        [ -f "$f" ] || continue
        printf '\n----- %s -----\n' "$f"
        tail -n 100 "$f" || true
    done
}
trap dump_logs ERR

# ---------------------------------------------------------------------------
# Prerequisites. `docker/slurm-ci/Dockerfile` bakes all of this in; NOTHING is
# installed here. An `up` therefore does not depend on apt being reachable, and
# the environment it produces is a property of the image alone.
#
# THIS USED TO apt-get THE MISSING PACKAGES, on the theory that the fallback
# made the image "an optimisation rather than a prerequisite". It did neither.
# Both of the following were measured, same script, same day:
#
#   It never bought a usable cluster. On a plain `ubuntu:24.04` the install
#   succeeds and the node does reach idle — and then the very next step of
#   `slurm_cluster.sh up` dies with `fake_serving_venv.sh: line 35: python3:
#   command not found`, because the interpreter, this project's pinned
#   dependencies, pytest and PYTHONPATH=/workspace come from the image too and
#   the fallback installed none of them. `probe` and `test` need all four. The
#   image was always a prerequisite; apt only delayed finding that out.
#
#   It silently contradicted the configuration written below. apt serves
#   whatever the base image's archive has, and this script's config is
#   version-specific: `CgroupPlugin=autodetect` + `IgnoreSystemd=yes` is the
#   23.11 spelling (see cgroup.conf, which spells out why `disabled` is wrong
#   before 24.05 and right after). On `ubuntu:26.04` the same fallback installs
#   slurm-wlm 25.11.2, writes the 23.11 cgroup.conf anyway, and slurmd dies —
#
#     error: cannot setup the scope for cgroup
#     error: Unable to initialize cgroup plugin
#     error: slurmd initialization failed
#
#   — surfacing five minutes later as "node is 'unknown*', not 'idle'" with the
#   cause buried in slurmd.log. A bring-up whose Slurm version is decided by the
#   caller's base image is the local-versus-CI divergence this cluster exists to
#   eliminate, arriving through the same back door as host-measured capacity.
#
# So: check, name what is missing, and say how to repair it. A missing package
# means the wrong image, and the only correct repair is to rebuild the image.
# ---------------------------------------------------------------------------
_say "Checking prerequisites"
# Two columns: what the bring-up invokes, and the apt package the Dockerfile
# installs to supply it. Entries are spelled exactly as they are invoked —
# absolute for the daemons, bare for anything resolved through PATH — so this
# checks the literal commands below rather than a curated approximation.
# squeue/scancel are not used here but are what the suite drives; they ship in
# the same package, so requiring them costs nothing and fails no sooner.
missing_pkgs=""
while read -r binary package; do
    [ -n "$binary" ] || continue
    if [ "${binary#/}" != "$binary" ]; then
        [ -x "$binary" ] && continue
    else
        command -v "$binary" >/dev/null 2>&1 && continue
    fi
    echo "  missing: ${binary}  (apt package: ${package})" >&2
    case " ${missing_pkgs} " in
        *" ${package} "*) ;;
        *) missing_pkgs="${missing_pkgs}${missing_pkgs:+ }${package}" ;;
    esac
done <<'REQUIRED'
sbatch                  slurm-wlm
squeue                  slurm-wlm
scancel                 slurm-wlm
sinfo                   slurm-wlm
sacct                   slurm-wlm
sacctmgr                slurm-wlm
scontrol                slurm-wlm
/usr/sbin/slurmctld     slurm-wlm
/usr/sbin/slurmd        slurm-wlm
/usr/sbin/slurmdbd      slurmdbd
/usr/sbin/munged        munge
munge                   munge
unmunge                 munge
/usr/sbin/mariadbd      mariadb-server
mariadb-install-db      mariadb-server
mysql                   mariadb-server
mysqladmin              mariadb-server
setpriv                 util-linux
pgrep                   procps
REQUIRED

if [ -n "$missing_pkgs" ]; then
    cat >&2 <<MSG

FAIL: this is not the cluster image — the commands listed above are missing.
      Missing apt packages: ${missing_pkgs}

      They are installed by docker/slurm-ci/Dockerfile and are deliberately NOT
      installed at run time: the Slurm configuration this script writes is
      specific to the version the image ships, so a cluster assembled from
      whatever apt happens to serve would be configured wrong. Rebuild:

          scripts/slurm_cluster.sh rebuild

      If EVAL360_SLURM_IMAGE is set, unset it or point it back at
      eval360-slurm-ci:local. If you added a dependency, add it to the
      Dockerfile's apt-get line and to the table above, not here.
MSG
    exit 1
fi

# dpkg only. The Slurm CLIs are NOT probed here, even for `--version`: since
# 23.11 a client with no slurm.conf falls back to configless mode, attempts a
# DNS SRV lookup for the controller, and exits fatal —
#
#   sbatch: error: resolve_ctls_from_dns_srv: res_nsearch error: Unknown host
#   sbatch: fatal: Could not establish a configuration source
#
# — so `sbatch --version` before the config is written kills the bring-up. The
# tool versions are reported after configuration instead.
dpkg-query -W -f='slurm-wlm  = ${Version}\n' slurm-wlm
dpkg-query -W -f='slurmdbd   = ${Version}\n' slurmdbd

# ---------------------------------------------------------------------------
# munge — every other daemon authenticates through it, so it goes first and its
# failure must be distinguishable from a Slurm failure.
# ---------------------------------------------------------------------------
_say "Starting munge"
install -d -o munge -g munge -m 0700 /etc/munge /var/lib/munge /var/log/munge
install -d -o munge -g munge -m 0755 /run/munge
if [ ! -f /etc/munge/munge.key ]; then
    dd if=/dev/urandom bs=1 count=1024 status=none of=/etc/munge/munge.key
    chown munge:munge /etc/munge/munge.key
    chmod 400 /etc/munge/munge.key
fi
pgrep -x munged >/dev/null 2>&1 || setpriv --reuid=munge --regid=munge --init-groups /usr/sbin/munged --force
munge -n | unmunge | head -n 5

# ---------------------------------------------------------------------------
# The accounting database. Without it sacct answers nothing.
#
# Started directly rather than via `service`/systemctl, which do not exist here.
# mariadb-install-db is conditional because the package postinst cannot run it
# during a container build.
# ---------------------------------------------------------------------------
_say "Starting MariaDB"
install -d -o mysql -g mysql -m 0755 /run/mysqld /var/lib/mysql
if [ ! -d /var/lib/mysql/mysql ]; then
    mariadb-install-db --user=mysql --datadir=/var/lib/mysql >/dev/null
fi
if ! mysqladmin ping --silent 2>/dev/null; then
    nohup /usr/sbin/mariadbd --user=mysql >/var/log/mariadbd.log 2>&1 &
    for _ in $(seq 1 60); do
        mysqladmin ping --silent 2>/dev/null && break
        sleep 1
    done
fi
mysqladmin ping

mysql <<SQL
CREATE DATABASE IF NOT EXISTS slurm_acct_db;
CREATE USER IF NOT EXISTS 'slurm'@'localhost' IDENTIFIED BY '${DB_PASSWORD}';
GRANT ALL ON slurm_acct_db.* TO 'slurm'@'localhost';
FLUSH PRIVILEGES;
SQL
echo "slurm_acct_db ready"

# ---------------------------------------------------------------------------
# Configuration.
#
# SlurmdParameters=config_overrides is what makes the fiction possible: it tells
# slurmd to accept the configured hardware rather than reject the node when its
# own inventory disagrees. Without it a node claiming GPUs it does not have
# registers INVAL and never schedules anything.
# ---------------------------------------------------------------------------
_say "Writing configuration"
install -d -o slurm -g slurm -m 0755 "$CONF_DIR" "$LOG_DIR" /var/spool/slurmctld
install -d -o root  -g root  -m 0755 /var/spool/slurmd
# slurmdbd runs as the slurm user and cannot write a pidfile into the root-owned
# /run tmpfs ("Unable to open pidfile `/run/slurmdbd.pid': Permission denied").
# Give it a directory it owns rather than running it as root.
install -d -o slurm -g slurm -m 0755 /run/slurm

tee "$CONF_DIR/slurm.conf" >/dev/null <<CONF
ClusterName=${CLUSTER_NAME}
SlurmctldHost=${HOST}

SlurmUser=slurm
SlurmdUser=root
AuthType=auth/munge

StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldPidFile=/run/slurmctld.pid
SlurmdPidFile=/run/slurmd.pid
SlurmctldLogFile=${LOG_DIR}/slurmctld.log
SlurmdLogFile=${LOG_DIR}/slurmd.log

# proctrack/linuxproc and task/none on purpose: the cgroup plugins need
# privileges this container deliberately does not ask for. See cgroup.conf
# below — choosing non-cgroup plugins here is NOT sufficient on its own.
ProctrackType=proctrack/linuxproc
TaskPlugin=task/none
SchedulerType=sched/backfill

# Neither is used here: nothing sends mail and nothing runs MPI. MailProg stops
# a real "Configured MailProg is invalid" error. MpiDefault=none does NOT silence
# the "mpi/pmix: can not load PMIx library" errors slurmctld logs at startup —
# (double quotes, not backticks: this heredoc is unquoted so that \${HOST} and
# friends expand, which means backticks would be COMMAND-SUBSTITUTED — the words
# between them vanished from the written config and bash tried to run them)
# it enumerates every installed MPI plugin regardless of the default — so those
# lines are expected in the log and are not a symptom of anything.
MailProg=/bin/true
MpiDefault=none
SelectType=select/cons_tres
SelectTypeParameters=CR_Core
ReturnToService=2

# The node is declared with hardware it does not have, and slurmd is told to
# believe the declaration rather than its own inventory.
GresTypes=gpu
SlurmdParameters=config_overrides

# Accounting through slurmdbd, which is what gives sacct a root row to
# return. AccountingStorageEnforce is deliberately unset (= no enforcement), so
# jobs are recorded without needing user/account associations to exist.
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=localhost
JobAcctGatherType=jobacct_gather/none

# Topology is spelled out rather than inferred: with SelectTypeParameters=
# CR_Core, Slurm allocates whole CORES, so a host that reports 2 CPUs as
# 1 core x 2 threads would fit only one job per core no matter what CPUs
# says. One thread per core keeps "one CPU" and "one core" the same thing.
NodeName=${HOST} CPUs=${FAKE_CPUS} Boards=1 SocketsPerBoard=1 CoresPerSocket=${FAKE_CPUS} ThreadsPerCore=1 RealMemory=${FAKE_MEM_MB} Gres=gpu:${FAKE_GPUS} State=UNKNOWN
PartitionName=ci Nodes=ALL Default=YES MaxTime=INFINITE State=UP
CONF

# A count-only GPU GRES does NOT work; fake device files are required.
#
# The obvious form — `NodeName=x Name=gpu Count=1`, no File — is accepted by
# slurmctld and then silently discarded by slurmd:
#
#   warning: Ignoring file-less GPU gpu:(null) from final GRES list
#
# `gpu` is special-cased. A file-less GRES is legal for a generic countable
# resource, but not for GPUs: slurmd drops it, registers the node with gpu:0
# against a configured gpu:1, and the node lands in INVALID_REG — which presents
# as a node stuck `inval` and every job PENDING forever, with the real cause one
# warning line deep in slurmd.log.
#
# So the GPUs get device files. They do not have to be GPUs — nothing opens
# them; Slurm only requires the paths to exist so it can build a device list.
# mknod with the nvidia major is cosmetic, chosen so the paths read as what they
# are standing in for. This needs --privileged, which the container already has
# for cgroupfs.
_say "Creating ${FAKE_GPUS} fake GPU device file(s)"
gpu_files=""
for i in $(seq 0 $((FAKE_GPUS - 1))); do
    device="/dev/eval360gpu${i}"
    if [ ! -e "$device" ]; then
        # 195 is the nvidia major. Any existing path would satisfy Slurm; if
        # mknod is refused, fall back to a plain file rather than failing the
        # bring-up, because Slurm does not actually open these.
        mknod "$device" c 195 "$i" 2>/dev/null || : > "$device"
    fi
    gpu_files="${gpu_files}${gpu_files:+,}${device}"
done
ls -l /dev/eval360gpu*

tee "$CONF_DIR/gres.conf" >/dev/null <<CONF
# AutoDetect=off keeps slurmd from trying NVML and failing on a machine with no
# driver. File= is mandatory for GPUs — see the note in slurm_up.sh.
AutoDetect=off
NodeName=${HOST} Name=gpu File=${gpu_files}
CONF

# IgnoreSystemd=yes, and this file is NOT optional.
#
# Choosing proctrack/linuxproc and task/none above does not stop slurmd from
# initialising a cgroup context. On a cgroup/v2 host, Slurm 23.11 wants to place
# slurmstepd in its own systemd scope and asks systemd to create it over dbus. A
# container has neither, so slurmd dies during startup —
#
#   error: cgroup_dbus_attach_to_scope: cannot connect to dbus system daemon:
#          Failed to connect to socket /run/dbus/system_bus_socket
#   error: cannot create cgroup context for cgroup/v2
#   error: Unable to initialize cgroup plugin
#   error: slurmd initialization failed
#
# — with slurmctld still up, so the symptom is a node that never registers
# rather than an obvious crash.
#
# IgnoreSystemd=yes is the option Slurm documents for this case: slurmd manages
# the cgroup directories itself instead of delegating to systemd. Note what does
# NOT work on 23.11: `CgroupPlugin=disabled` is a 24.05 addition, and on 23.11 it
# is taken as a plugin NAME, producing the equally fatal
#
#   error: Couldn't find the specified plugin name for disabled
#   error: cannot create cgroup context for disabled
#
# so if this image is ever moved to a newer Slurm, `disabled` becomes the
# simpler choice — but it must not be used before then.
tee "$CONF_DIR/cgroup.conf" >/dev/null <<CONF
CgroupPlugin=autodetect
IgnoreSystemd=yes
CONF

tee "$CONF_DIR/slurmdbd.conf" >/dev/null <<CONF
AuthType=auth/munge
DbdHost=localhost
DbdPort=6819
SlurmUser=slurm
DebugLevel=info
LogFile=${LOG_DIR}/slurmdbd.log
PidFile=/run/slurm/slurmdbd.pid
StorageType=accounting_storage/mysql
StorageHost=localhost
StorageUser=slurm
StoragePass=${DB_PASSWORD}
StorageLoc=slurm_acct_db
CONF
# slurmdbd refuses to start if this is readable by anyone else — it holds the
# database password.
chown slurm:slurm "$CONF_DIR/slurmdbd.conf"
chmod 600 "$CONF_DIR/slurmdbd.conf"

echo "--- slurm.conf ---"; cat "$CONF_DIR/slurm.conf"
echo "--- gres.conf ---";  cat "$CONF_DIR/gres.conf"
echo "--- cgroup.conf ---"; cat "$CONF_DIR/cgroup.conf"

# Safe now that slurm.conf exists — see the note in "Checking packages" above.
echo "--- tool versions ---"
sbatch --version
sacct --version

# ---------------------------------------------------------------------------
# Daemons, in dependency order. slurmdbd must be answering before slurmctld
# starts, or slurmctld comes up with accounting disabled and every sacct test
# gets a false negative that looks like a parser problem.
# ---------------------------------------------------------------------------
_say "Starting slurmdbd"
pgrep -x slurmdbd >/dev/null 2>&1 || setpriv --reuid=slurm --regid=slurm --init-groups /usr/sbin/slurmdbd
for _ in $(seq 1 30); do
    sacctmgr -i list cluster >/dev/null 2>&1 && break
    sleep 1
done
sacctmgr -i list cluster
sacctmgr -i add cluster "${CLUSTER_NAME}" 2>/dev/null || true

_say "Starting slurmctld and slurmd"
# slurmd creates /sys/fs/cgroup/system.slice/<node>_slurmstepd.scope but not the
# system.slice parent, which a container does not have. Needs cgroupfs writable,
# i.e. the --privileged in scripts/slurm_cluster.sh.
if [ -w /sys/fs/cgroup ]; then
    mkdir -p /sys/fs/cgroup/system.slice
else
    echo "WARNING: /sys/fs/cgroup is read-only; slurmd will fail to start." >&2
    echo "         The container needs --privileged (see scripts/slurm_cluster.sh)." >&2
fi
pgrep -x slurmctld >/dev/null 2>&1 || /usr/sbin/slurmctld
pgrep -x slurmd    >/dev/null 2>&1 || /usr/sbin/slurmd

# ---------------------------------------------------------------------------
# Does the node actually reach `idle`?
#
# This loop is the difference between "the daemons started" and "the cluster
# works". A node that registers INVAL or DOWN leaves the daemons running and
# every job PENDING forever, which without this check looks like a hang in
# whatever runs next rather than a cluster that never came up.
# ---------------------------------------------------------------------------
_say "Waiting for the node to reach idle"
resumed=0
for _ in $(seq 1 60); do
    state="$(sinfo -h -o '%T' -n "$HOST" 2>/dev/null | head -n 1 || true)"
    case "$state" in
        idle) break ;;
        inval*)
            # INVALID_REG cannot be cleared by RESUME — slurmctld rejects the
            # transition outright ("Invalid node state transition requested
            # ... from=INVAL to=RESUME"), so retrying just spams the log for two
            # minutes and then fails with the state rather than the cause. It
            # means slurmd registered hardware that does not match slurm.conf;
            # the reason is in slurmd.log, and for GPUs it is almost always the
            # file-less GRES trap described above.
            echo "FAIL: node ${HOST} registered as INVALID_REG." >&2
            echo "      slurmd's hardware does not match slurm.conf. Reason:" >&2
            grep -iE 'gres|gpu|invalid|error' "$LOG_DIR/slurmd.log" | tail -20 >&2
            exit 1
            ;;
        down*|drain*)
            # RESUME once: a node can register DOWN simply because it has no
            # prior state. If it goes back down, that is a real failure, so do
            # not keep poking it. (No Reason= — that is only meaningful when
            # setting DOWN or DRAIN.)
            if [ "$resumed" -eq 0 ]; then
                scontrol update NodeName="$HOST" State=RESUME || true
                resumed=1
            fi
            ;;
    esac
    sleep 2
done

sinfo -N -l
scontrol show node "$HOST"

state="$(sinfo -h -o '%T' -n "$HOST" | head -n 1)"
if [ "$state" != "idle" ]; then
    echo "FAIL: node ${HOST} is '${state}', not 'idle' — the cluster did not come up" >&2
    exit 1
fi

trap - ERR
_say "Cluster up: ${CLUSTER_NAME} / node ${HOST} / ${FAKE_CPUS} CPUs / ${FAKE_MEM_MB}MB / gpu:${FAKE_GPUS} (all declared, not measured)"
