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
REPO_URL="${HORIZON_REPO:-https://github.com/AmRitJain0442/horizon-uncertainty-rl.git}"

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

say "Generating a dashboard password"
DASH_PASS=$(openssl rand -base64 18)
DASH_HASH=$(docker run --rm caddy:2-alpine caddy hash-password --plaintext "$DASH_PASS")

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
git clone --depth 1 ${REPO_URL} /opt/horizon
cat > /opt/horizon/infra/.env <<ENV_EOF
POSTGRES_PASSWORD=$(openssl rand -hex 16)
HORIZON_RUN_ID=${HORIZON_RUN_ID:-t1-1}
HORIZON_PROJECT=${PROJECT}
HORIZON_WORKERS=${HORIZON_WORKERS:-24}
HORIZON_EPISODES=${HORIZON_EPISODES:-200}
HORIZON_EXIT=${HORIZON_EXIT:-uniform}
HORIZON_INFO=${HORIZON_INFO:-K}
DASH_HOST=${DASH_HOST}
DASH_USER=horizon
DASH_HASH=${DASH_HASH}
ENV_EOF

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
ExecStart=/usr/bin/docker compose up -d --build
ExecStop=/usr/bin/docker compose down
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
UNIT_EOF

systemctl daemon-reload
systemctl enable --now horizon.service
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
  Username    horizon
  Password    ${DASH_PASS}

  Save that password now; it is not stored anywhere else.
  First boot takes a few minutes while Docker builds the image, and Caddy needs
  a moment more to obtain its certificate.

  Logs        gcloud compute ssh ${VM} --zone ${ZONE} --project ${PROJECT} \\
                --command 'sudo docker compose -f /opt/horizon/infra/docker-compose.yml logs -f'
  Stop        gcloud compute instances stop ${VM} --zone ${ZONE} --project ${PROJECT}

SUMMARY
