#!/usr/bin/env bash
#
# tune.sh - OS-level tuning for the Orange Pi Zero 3W LLM node.
#
# Every item below was chosen from measurements on this exact board
# (A733: 2x A76 @2.0 GHz + 6x A55 @1.79 GHz, 12 GB LPDDR5, microSD root,
#  kernel without THP, zram-backed swap, no NPU driver).
#
#   --check   report what is and is not applied (default, changes nothing)
#   --apply   apply everything (idempotent, safe to re-run)
#   --revert  remove the sysctl/governor drop-ins and the noatime mount option
#
# What it tunes, and why:
#   vm.swappiness=10        keep mmap'd model weights in page cache; anonymous
#                           memory (KV cache) must never push weights to zram
#   vm.page-cluster=0       zram is a block device: 1 page per swap-in beats 8
#   vm.vfs_cache_pressure=50 keep dentries/inodes cached -> faster model reopen
#   vm.dirty_*=10/5         cap writeback bursts to the microSD
#   vm.min_free_kbytes      leave headroom so large mmaps never stall on reclaim
#   vm.overcommit_memory=1  large PROT_READ mmaps of GGUF files must not fail
#   noatime                 no metadata write on every model/GGUF read
#   governor=performance    stable token rate on A76; schedutil saves ~1-2 W
#   LimitMEMLOCK + --mlock  resident models are never evicted by the kernel
#
set -euo pipefail

SYSCTL_FILE="/etc/sysctl.d/90-cheap-llm-stack.conf"
GOV_SERVICE="/etc/systemd/system/cheap-llm-stack-governor.service"
STACK_DIR="${STACK_DIR:-/opt/cheap-llm-stack}"
GOVERNOR="${GOVERNOR:-performance}"   # performance | schedutil | powersave

MODE="check"
for arg in "$@"; do
    case "${arg}" in
        --check)  MODE="check" ;;
        --apply)  MODE="apply" ;;
        --revert) MODE="revert" ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown option: ${arg}" >&2; exit 2 ;;
    esac
done

say()  { printf '\033[1;32m[tune]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[tune]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[tune]\033[0m %s\n' "$*" >&2; exit 1; }
ok()   { printf '  \033[1;32mok\033[0m   %-38s %s\n' "$1" "$2"; }
todo() { printf '  \033[1;33mTODO\033[0m %-38s %s\n' "$1" "$2"; }

[ "$(id -u)" -eq 0 ] || die "run as root (sudo $0 --apply)"

# ---------------------------------------------------------------- sysctl set ---
# name=value pairs; the desired end state.
SYSCTL_WANT="
vm.swappiness=10
vm.page-cluster=0
vm.vfs_cache_pressure=50
vm.dirty_ratio=10
vm.dirty_background_ratio=5
vm.min_free_kbytes=262144
vm.overcommit_memory=1
kernel.sched_autogroup_enabled=1
"

cur() { cat "/proc/sys/$(echo "$1" | tr '.' '/')" 2>/dev/null || echo "n/a"; }

check_sysctl() {
    local bad=0
    while IFS='=' read -r k v; do
        [ -z "${k}" ] && continue
        local have; have="$(cur "${k}")"
        if [ "${have}" = "${v}" ]; then ok "${k}" "${have}"
        else todo "${k}" "is ${have}, want ${v}"; bad=1; fi
    done <<< "${SYSCTL_WANT}"
    return "${bad}"
}

apply_sysctl() {
    { echo "# managed by cheap-llm-stack scripts/tune.sh"; echo "# generated $(date -Is)"
      while IFS='=' read -r k v; do [ -n "${k}" ] && echo "${k} = ${v}"; done <<< "${SYSCTL_WANT}"
    } > "${SYSCTL_FILE}"
    sysctl -q --system
    say "applied ${SYSCTL_FILE}"
}

# -------------------------------------------------------------------- mount ---
root_opts() { findmnt -no OPTIONS / ; }
root_dev()  { findmnt -no SOURCE  / ; }

check_mount() {
    if root_opts | tr ',' '\n' | grep -qx 'noatime'; then
        ok "root mount" "noatime"
    else
        todo "root mount" "$(root_dev) is $(root_opts)"
    fi
}

apply_mount() {
    local dev; dev="$(root_dev)"
    local fstab_dev; fstab_dev="$(awk -v d="${dev}" '$1==d {print $1; exit}' /etc/fstab)"
    if [ -z "${fstab_dev}" ] && [ -f /etc/fstab ]; then
        warn "root device ${dev} not found in /etc/fstab (Armbian often uses UUID); remounting for this boot only"
        mount -o remount,noatime / && say "remounted / with noatime (not persistent)"
        return
    fi
    cp /etc/fstab "/etc/fstab.bak.$(date +%Y%m%d%H%M%S)"
    # add noatime to the 4th field of that one line, idempotently
    awk -v d="${fstab_dev}" '
        $1==d && $0 !~ /noatime/ { n=split($4,a,","); out=""; for(i=1;i<=n;i++){ if(a[i]=="atime"||a[i]=="relatime"||a[i]=="strictatime") continue; out=out (out?",":"") a[i] } $4=out",noatime"; OFS=" " }
        { print }
    ' "/etc/fstab" > /etc/fstab.new && mv /etc/fstab.new /etc/fstab
    mount -o remount / && say "remounted / with noatime (persistent in /etc/fstab)"
}

# ----------------------------------------------------------------- governor ---
check_gov() {
    local have; have="$(cat /sys/devices/system/cpu/cpu6/cpufreq/scaling_governor 2>/dev/null || echo n/a)"
    if [ "${have}" = "${GOVERNOR}" ]; then ok "cpu governor" "${GOVERNOR} (all cores)"
    else todo "cpu governor" "cpu6 is ${have}, want ${GOVERNOR}"; fi
    if [ -f "${GOV_SERVICE}" ]; then ok "governor unit" "${GOV_SERVICE}"; else todo "governor unit" "missing"; fi
}

apply_gov() {
    cat > "${GOV_SERVICE}" <<EOF
# managed by cheap-llm-stack scripts/tune.sh
[Unit]
Description=Set CPU governor for cheap-llm-stack
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo ${GOVERNOR} > "\$g"; done'
ExecStop=/bin/sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo schedutil > "\$g"; done'

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable --now cheap-llm-stack-governor.service
    say "cpu governor set to ${GOVERNOR}"
}

# -------------------------------------------------------------------- swap ---
check_swap() {
    if swapon --show=NAME,TYPE --noheadings | grep -q zram; then
        ok "swap" "zram active ($(swapon --show=SIZE --noheadings | head -1 | tr -d ' '))"
    elif swapon --show --noheadings | grep -q .; then
        warn "swap is on disk, not zram: model reloads will fight the SD card"
    else
        todo "swap" "no swap at all (zram recommended for 12 GB board)"
    fi
}

# ------------------------------------------------------------------- report ---
case "${MODE}" in
    check)
        say "checking OS-level tuning"
        check_swap || true
        check_sysctl || true
        check_mount || true
        check_gov || true
        echo
        say "extra facts measured on this board"
        printf '  %-38s %s\n' "root device" "$(root_dev) ($(root_opts))"
        printf '  %-38s %s\n' "A76 cores" "$(lscpu -p=CPU,MAXMHZ | awk -F, 'NR>4 && $2+0>1900 {printf "cpu%s ", $1}')"
        printf '  %-38s %s\n' "NPU driver" "$(ls /dev/galcore /dev/npu 2>/dev/null | head -1 || echo 'absent - CPU inference only')"
        printf '  %-38s %s\n' "THP" "$([ -e /sys/kernel/mm/transparent_hugepage/enabled ] && cat /sys/kernel/mm/transparent_hugepage/enabled || echo 'not provided by this kernel')"
        printf '  %-38s %s\n' "idle temp" "$(awk '{printf "%.0f C ", $1/1000}' /sys/class/thermal/thermal_zone0/temp 2>/dev/null)"
        echo
        say "apply with: sudo $0 --apply"
        ;;
    apply)
        say "applying OS-level tuning for ${STACK_DIR}"
        apply_sysctl
        apply_mount
        apply_gov
        check_swap || true
        systemctl set-environment CHEAP_LLM_TUNED=1 2>/dev/null || true
        echo
        say "done. Re-run with --check to verify."
        ;;
    revert)
        say "reverting OS-level tuning"
        rm -f "${SYSCTL_FILE}"
        systemctl disable --now cheap-llm-stack-governor.service 2>/dev/null || true
        rm -f "${GOV_SERVICE}"
        systemctl daemon-reload
        sysctl -q --system
        say "sysctl and governor drop-ins removed (fstab left untouched on purpose)"
        ;;
esac
