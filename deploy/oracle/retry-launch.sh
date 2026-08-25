#!/usr/bin/env bash
#
# Retry launching the Always Free A1 instance until capacity appears.
#
# "Out of capacity for shape VM.Standard.A1.Flex" is not a misconfiguration --
# by the time you see it the console has already validated everything else. It
# means every eligible host in that availability domain is full right now.
# Capacity is released continuously as other tenants terminate instances, and
# the window is often seconds wide, which is why clicking Create by hand can
# fail for days while a loop succeeds overnight.
#
# WHERE TO RUN THIS
#
#   OCI Cloud Shell (the >_ icon in the console header). The OCI CLI is
#   preinstalled and already authenticated as you, so there is no API key to
#   generate and no config to write. Paste this file in with the editor, or
#   `cat > retry-launch.sh` and paste.
#
#   Cloud Shell disconnects after roughly 20 minutes idle. The loop dies with
#   the session, so start it under tmux and detach:
#
#       tmux new -s launch
#       ./retry-launch.sh
#       # Ctrl-B then D to detach; `tmux attach -t launch` to come back
#
#   Even then Cloud Shell has a maximum session lifetime. For a multi-day hunt,
#   install the CLI locally (`oci setup config`) and run it there instead.
#
# BE POLITE. SLEEP_SECONDS=60 is deliberate. Hammering the API every second
# will not find capacity sooner -- allocation is not first-come-polling -- and
# it can get you throttled, which makes things strictly worse.

set -uo pipefail

# ---------------------------------------------------------------- settings --

DISPLAY_NAME="${DISPLAY_NAME:-Palate}"
SHAPE="VM.Standard.A1.Flex"

# 1 OCPU / 6GB is the A1.Flex minimum and the easiest thing to place. The free
# allotment is 4 OCPU / 24GB, but ask for that now and you compete for a much
# scarcer host. Take the smallest instance that runs the service, then resize
# later -- possible precisely because Shielded Instance was left off, which
# would have frozen the shape at launch.
OCPUS="${OCPUS:-1}"
MEM_GB="${MEM_GB:-6}"

# Ubuntu 24.04 by default so deploy/oracle/README.md applies verbatim: its
# steps assume the `ubuntu` user, apt, and iptables. Oracle Linux 9 works just
# as well but wants `opc`, dnf and firewall-cmd instead. Image choice has no
# effect on capacity, so pick whichever you would rather administer.
OS_NAME="${OS_NAME:-Canonical Ubuntu}"
OS_VERSION="${OS_VERSION:-24.04}"

SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/id_ed25519.pub}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"   # 0 = forever

# --------------------------------------------------------------- discovery --

echo "== discovering resources =="

COMPARTMENT_OCID="${COMPARTMENT_OCID:-${OCI_TENANCY:-}}"
if [ -z "$COMPARTMENT_OCID" ]; then
	COMPARTMENT_OCID=$(oci iam compartment list --include-root \
		--query "data[?contains(id,'tenancy')].id | [0]" --raw-output 2>/dev/null)
fi
[ -z "$COMPARTMENT_OCID" ] && { echo "FATAL: set COMPARTMENT_OCID manually"; exit 1; }
echo "compartment : $COMPARTMENT_OCID"

# The PUBLIC subnet. If you have more than one, set SUBNET_OCID yourself --
# picking the private one produces an instance with no route to the internet,
# which looks healthy and answers nothing.
if [ -z "${SUBNET_OCID:-}" ]; then
	SUBNET_OCID=$(oci network subnet list -c "$COMPARTMENT_OCID" \
		--query "data[?\"prohibit-public-ip-on-vnic\"==\`false\`].id | [0]" --raw-output 2>/dev/null)
fi
[ -z "$SUBNET_OCID" ] && { echo "FATAL: no public subnet found; set SUBNET_OCID"; exit 1; }
echo "subnet      : $SUBNET_OCID"

# Must be an aarch64 image. Filtering by --shape makes the API return only
# images that can actually boot on A1, so an x86 build cannot be selected here.
if [ -z "${IMAGE_OCID:-}" ]; then
	IMAGE_OCID=$(oci compute image list -c "$COMPARTMENT_OCID" \
		--operating-system "$OS_NAME" --operating-system-version "$OS_VERSION" \
		--shape "$SHAPE" --sort-by TIMECREATED --sort-order DESC \
		--query "data[0].id" --raw-output 2>/dev/null)
fi
[ -z "$IMAGE_OCID" ] && { echo "FATAL: no $OS_NAME $OS_VERSION image for $SHAPE"; exit 1; }
echo "image       : $IMAGE_OCID  ($OS_NAME $OS_VERSION)"

if [ ! -f "$SSH_KEY_FILE" ]; then
	echo "FATAL: no public key at $SSH_KEY_FILE"
	echo "In Cloud Shell, recreate it with:"
	echo "  mkdir -p ~/.ssh && echo 'ssh-ed25519 AAAA...your-key... palate-oracle' > $SSH_KEY_FILE"
	exit 1
fi
echo "ssh key     : $SSH_KEY_FILE"

# Every AD in the region. A REGIONAL subnet can host an instance in any of
# them, so trying each per round multiplies your chances at no cost. Regions
# with a single AD simply yield a one-element list.
mapfile -t ADS < <(oci iam availability-domain list --query 'data[*].name' --raw-output 2>/dev/null | tr -d '[]," ' | grep -v '^$')
[ "${#ADS[@]}" -eq 0 ] && { echo "FATAL: could not list availability domains"; exit 1; }
echo "ADs         : ${ADS[*]}"
echo

# -------------------------------------------------------------------- loop --

attempt=0
while :; do
	attempt=$((attempt + 1))
	[ "$MAX_ATTEMPTS" -gt 0 ] && [ "$attempt" -gt "$MAX_ATTEMPTS" ] && {
		echo "giving up after $MAX_ATTEMPTS attempts"; exit 1; }

	for AD in "${ADS[@]}"; do
		printf '[%s] attempt %d -- %s ... ' "$(date +%H:%M:%S)" "$attempt" "$AD"

		# No --fault-domain on purpose: Oracle's own error message suggests
		# omitting it, because pinning one narrows the eligible host pool
		# exactly when the pool is the problem.
		ERR=$(oci compute instance launch \
			--compartment-id "$COMPARTMENT_OCID" \
			--availability-domain "$AD" \
			--display-name "$DISPLAY_NAME" \
			--shape "$SHAPE" \
			--shape-config "{\"ocpus\":$OCPUS,\"memoryInGBs\":$MEM_GB}" \
			--image-id "$IMAGE_OCID" \
			--subnet-id "$SUBNET_OCID" \
			--assign-public-ip true \
			--ssh-authorized-keys-file "$SSH_KEY_FILE" \
			--wait-for-state RUNNING \
			2>&1)

		if [ $? -eq 0 ]; then
			echo "LAUNCHED"
			echo
			echo "== public IP =="
			oci compute instance list-vnics --instance-id \
				"$(echo "$ERR" | grep -oE 'ocid1\.instance\.[a-z0-9._-]+' | head -1)" \
				--query 'data[0]."public-ip"' --raw-output
			echo
			echo "Next: deploy/oracle/README.md step 4 onward."
			exit 0
		fi

		# Retry ONLY on capacity. Anything else -- a bad OCID, a service limit,
		# a malformed shape config -- will fail identically forever, and a loop
		# that swallows it wastes hours hiding a one-line fix.
		if echo "$ERR" | grep -qiE 'out of capacity|outofhostcapacity|insufficient capacity'; then
			echo "no capacity"
		else
			echo "FAILED (not a capacity error) -- stopping"
			echo
			echo "$ERR"
			exit 1
		fi
	done

	sleep "$SLEEP_SECONDS"
done
