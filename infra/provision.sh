#!/usr/bin/env bash
# Provision the always-on VM that runs the experiment and serves the dashboard.
#
# Idempotent: safe to re-run. Creates the static IP, uploads the service-account
# key to Secret Manager, creates the VM with a startup script that pulls the
# secret and brings the stack up under systemd, and prints the dashboard URL.
#
# The researcher's laptop is not in the loop after this returns.

set -euo pipefail

PROJECT="${HORIZON_PROJECT:-project-a2dcdad0-5d65-4d61-846}"
ZONE="${HORIZON_ZONE:-us-central1-a}"
REGION="${ZONE%-*}"
VM="${HORIZON_VM:-horizon-runner}"
MACHINE="${HORIZON_MACHINE:-e2-standard-4}"
ADDRESS="${VM}-ip"
SECRET="${HORIZON_SECRET:-horizon-sa-key}"
SA_KEY="${HORIZON_SA_KEY_FILE:?set HORIZON_SA_KEY_FILE to the service-account json path}"
BUCKET="${HORIZON_BUCKET:-${PROJECT}-horizon}"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

say "Enabling required APIs"
gcloud services enable compute.googleapis.com secretmanager.googleapis.com \
  aiplatform.googleapis.com --project "$PROJECT"

say "Reserving a static external IP"
gcloud compute addresses describe "$ADDRESS" --region "$REGION" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud compute addresses create "$ADDRESS" --region "$REGION" --project "$PROJECT"
IP=$(gcloud compute addresses describe "$ADDRESS" --region "$REGION" \
       --project "$PROJECT" --format='value(address)')
DASH_HOST="${IP//./-}.sslip.io"

say "Storing the service-account key in Secret Manager"
# The key never enters the repository or the image; it is pulled at boot into a
# tmpfs mount and disappears on reboot.
gcloud secrets describe "$SECRET" --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud secrets create "$SECRET" --replication-policy=automatic --project "$PROJECT"
gcloud secrets versions add "$SECRET" --data-file="$SA_KEY" --project "$PROJECT" >/dev/null

say "Resolving the database password"
# Postgres only sets its password when it initialises an EMPTY data directory.
# Generating a fresh one per deploy therefore breaks authentication the moment
# the pgdata volume survives a redeploy, so the password is created once and
# kept in Secret Manager.
PG_SECRET="${SECRET}-pg"
if ! gcloud secrets describe "$PG_SECRET" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud secrets create "$PG_SECRET" --replication-policy=automatic --project "$PROJECT" >/dev/null
  openssl rand -hex 16 | gcloud secrets versions add "$PG_SECRET" --data-file=-     --project "$PROJECT" >/dev/null
fi
PG_PASSWORD=$(gcloud secrets versions access latest --secret="$PG_SECRET" --project "$PROJECT")

say "Shipping the code to GCS"
# The repository is private, so the VM cannot clone it. Ship a tarball through a
# bucket instead: it needs no credentials beyond the VM's own service account,
# and it survives reboots because the startup script re-fetches it every boot.
gcloud storage buckets describe "gs://${BUCKET}" --project "$PROJECT" >/dev/null 2>&1   || gcloud storage buckets create "gs://${BUCKET}" --project "$PROJECT" --location "$REGION"
TARBALL=$(mktemp -u).tar.gz
tar --exclude='.git' --exclude='data/dryrun' --exclude='__pycache__'     --exclude='*.egg-info' --exclude='.pytest_cache'     --exclude='gen-lang-client-*.json'     -czf "$TARBALL" horizon infra pyproject.toml README.md data/calibration.json
gcloud storage cp "$TARBALL" "gs://${BUCKET}/horizon.tar.gz" --project "$PROJECT"
rm -f "$TARBALL"

say "Granting the VM service account access to the secret and the bucket"
# The VM runs as the project's default compute service account, which has no
# access to either by default. Without these grants the startup script aborts at
# the secret fetch and the stack never comes up.
VM_SA="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding "$SECRET"   --member="serviceAccount:${VM_SA}" --role=roles/secretmanager.secretAccessor   --project "$PROJECT" >/dev/null
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}"   --member="serviceAccount:${VM_SA}" --role=roles/storage.objectViewer   --project "$PROJECT" >/dev/null

say "Opening the firewall for HTTP/HTTPS"
gcloud compute firewall-rules describe horizon-web --project "$PROJECT" >/dev/null 2>&1 \
  || gcloud compute firewall-rules create horizon-web \
       --allow=tcp:80,tcp:443 --target-tags=horizon --project "$PROJECT"

STARTUP=$(mktemp)
cat > "$STARTUP" <<STARTUP_EOF
#!/usr/bin/env bash
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y ca-certificates curl git
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=\$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/debian \$(. /etc/os-release && echo \$VERSION_CODENAME) stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# tmpfs so the key never touches disk
mkdir -p /run/secrets
mountpoint -q /run/secrets || mount -t tmpfs -o size=1m,mode=0700 tmpfs /run/secrets
gcloud secrets versions access latest --secret=${SECRET} --project=${PROJECT} \
  > /run/secrets/sa-key.json
chmod 0400 /run/secrets/sa-key.json

rm -rf /opt/horizon
mkdir -p /opt/horizon
gcloud storage cp gs://${BUCKET}/horizon.tar.gz /tmp/horizon.tar.gz
tar -xzf /tmp/horizon.tar.gz -C /opt/horizon
rm -f /tmp/horizon.tar.gz
cat > /opt/horizon/infra/.env <<ENV_EOF
POSTGRES_PASSWORD=${PG_PASSWORD}
HORIZON_RUN_ID=${HORIZON_RUN_ID:-t1-1}
HORIZON_PROJECT=${PROJECT}
HORIZON_WORKERS=${HORIZON_WORKERS:-24}
HORIZON_EPISODES=${HORIZON_EPISODES:-200}
HORIZON_EXIT=${HORIZON_EXIT:-uniform}
HORIZON_INFO=${HORIZON_INFO:-K}
DASH_HOST=${DASH_HOST}
ENV_EOF

# Caddy config is written here rather than shipped, so the hostname can be
# The hostname is substituted when this startup script is generated on the
# workstation; the QUOTED heredoc delimiter stops the VM re-expanding it.
# No authentication: the page carries only synthetic experiment telemetry.
cat > /opt/horizon/infra/Caddyfile <<'CADDY_EOF'
${DASH_HOST} {
    encode gzip
    reverse_proxy dashboard:8000
    log {
        output stdout
        format console
    }
}
CADDY_EOF

cat > /etc/systemd/system/horizon.service <<UNIT_EOF
[Unit]
Description=Random-horizon trading experiment
Requires=docker.service
After=docker.service network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/horizon/infra
ExecStartPre=/bin/bash -c 'mountpoint -q /run/secrets || mount -t tmpfs -o size=1m,mode=0700 tmpfs /run/secrets; gcloud secrets versions access latest --secret=${SECRET} --project=${PROJECT} > /run/secrets/sa-key.json'
ExecStart=/bin/bash -c 'if [ "${HORIZON_RESET_DB:-0}" = "1" ]; then docker compose down -v || true; fi; docker compose up -d --build'
ExecStop=/usr/bin/docker compose down
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
UNIT_EOF

systemctl daemon-reload
systemctl enable --now horizon.service

# SSH from a Windows workstation is unreliable, so surface container state on the
# serial console where it can always be read.
( sleep 150
  echo "=== HORIZON DIAGNOSTICS ===" > /dev/console
  docker compose -f /opt/horizon/infra/docker-compose.yml ps > /dev/console 2>&1
  echo "=== CADDY ===" > /dev/console
  docker compose -f /opt/horizon/infra/docker-compose.yml logs --tail 20 caddy > /dev/console 2>&1
  echo "=== ORCHESTRATOR ===" > /dev/console
  docker compose -f /opt/horizon/infra/docker-compose.yml logs --tail 20 orchestrator > /dev/console 2>&1
  echo "=== END DIAGNOSTICS ===" > /dev/console
) &
STARTUP_EOF

say "Creating the VM"
if gcloud compute instances describe "$VM" --zone "$ZONE" --project "$PROJECT" >/dev/null 2>&1; then
  echo "instance exists; updating startup script and restarting"
  gcloud compute instances add-metadata "$VM" --zone "$ZONE" --project "$PROJECT" \
    --metadata-from-file startup-script="$STARTUP"
  gcloud compute instances reset "$VM" --zone "$ZONE" --project "$PROJECT"
else
  gcloud compute instances create "$VM" \
    --project "$PROJECT" --zone "$ZONE" --machine-type "$MACHINE" \
    --image-family=debian-12 --image-project=debian-cloud \
    --boot-disk-size=50GB --boot-disk-type=pd-balanced \
    --address "$IP" --tags horizon \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata-from-file startup-script="$STARTUP"
fi
rm -f "$STARTUP"

cat <<SUMMARY

  Dashboard   https://${DASH_HOST}

  Open to anyone holding the link: the page shows only synthetic experiment
  telemetry, no credentials and no personal data.
  First boot takes a few minutes while Docker builds the image, and Caddy needs
  a moment more to obtain its certificate.

  Logs        gcloud compute ssh ${VM} --zone ${ZONE} --project ${PROJECT} \\
                --command 'sudo docker compose -f /opt/horizon/infra/docker-compose.yml logs -f'
  Stop        gcloud compute instances stop ${VM} --zone ${ZONE} --project ${PROJECT}

SUMMARY
