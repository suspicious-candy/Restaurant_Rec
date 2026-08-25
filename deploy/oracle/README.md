# Migrating the recommender from Cloud Run to Oracle Always Free

Why: Cloud Run billed roughly **$1.04 net over two days** — a ~$16/month run
rate — almost none of it compute. The Cloud Run free tier (240,000 vCPU-s,
450,000 GiB-s, 2M requests per month) covered the actual request handling. What
it did not cover was **storage of repeated deploy artifacts**: a ~1.8GB image
where `COPY . .` bundled the 553MB index into a layer that changed on every
source edit, plus a `--source .` build context tarball of the same size landing
in a staging bucket each deploy. Artifact Registry gives 0.5GB free, then
$0.10/GB/month, and nothing garbage-collects the old copies.

Oracle's Always Free tier removes the cause rather than the symptom. A real disk
means the index leaves the image entirely.

**Verify the cause before trusting this number.** Billing → Cost breakdown, group
by service. If the spend is under *Cloud Run* rather than Artifact Registry or
Cloud Build, something is wrong with the service's configuration or traffic, and
that misconfiguration follows you onto a VM where it burns real cores. If it is
under *Cloud Build*, the spend was a burst from deploying ~10× on Aug 19–21 and
will decay on its own.

## What changes

| | Cloud Run | Oracle A1 |
|---|---|---|
| Idle behavior | Scales to zero, cold start | Always on |
| `chroma_db/` | Baked into the image, 553MB per deploy | Bind-mounted from disk |
| `POST /index/missing` | Writes vanish on scale-down | Persists |
| TLS | Provided | Caddy + Let's Encrypt |
| Cost | ~$16/month | $0 |
| You administer | Nothing | OS, firewall, certs, restarts |

The always-on property matters more than the money. `/api/Restaurants/nearby`
gives up after 3 seconds, so on Cloud Run the first dashboard load after an idle
period got an unranked list by design. That stops happening here.

## Files

- `../../Dockerfile.arm64` — ARM build, no index baked in
- `../../Dockerfile.arm64.dockerignore` — BuildKit per-Dockerfile context; the
  Cloud Run `.dockerignore` is untouched so rollback stays possible
- `docker-compose.yml` — recommender + Caddy
- `Caddyfile` — TLS termination and the 300s proxy timeout
- `.env.example` — copy to `.env` on the VM

---

## 1. Provision the VM

Oracle Cloud → Compute → Instances → **Create instance**.

- **Image:** Canonical Ubuntu 24.04 — make sure you pick the **aarch64** build
- **Shape:** Ampere A1 Flex, **2 OCPU / 12 GB**. The Always Free allotment is
  4 OCPU / 24 GB total; taking half leaves room for a second instance and is
  already six times the memory the Cloud Run revision had.
- **Boot volume:** 100 GB (Always Free covers 200 GB total)
- **SSH key:** upload your public key. There is no password login.

Note the **public IP**.

> **The likely blocker.** Always Free A1 capacity is heavily contested and
> "Out of host capacity" is common, especially in Mumbai and Hyderabad. If you
> hit it, try a different availability domain, or retry — capacity is released
> continuously. If it will not provision at all, Hugging Face Spaces is the
> fallback, at the cost of a 48-hour idle sleep and a 30–90s wake.

## 2. Know the idle-reclamation rule

Oracle reclaims Always Free compute that stays idle — roughly, under ~20% CPU,
network and memory utilisation across a 7-day window. A recommender that only
answers occasional queries can trip this. Nothing to do yet; note it, and see
step 11.

## 3. Open the Oracle firewall (layer 1 of 2)

VCN → Security Lists → **Default Security List** → Add Ingress Rules.

Two rules, both stateless=No, source CIDR `0.0.0.0/0`, IP protocol TCP:

- Destination port **80** — required for Let's Encrypt's HTTP-01 challenge
- Destination port **443** — the actual traffic

## 4. SSH in

```bash
ssh ubuntu@YOUR_PUBLIC_IP
```

## 5. Open the VM's firewall (layer 2 of 2)

**This is the step everyone forgets.** Oracle's Ubuntu images ship iptables
rules that REJECT inbound traffic regardless of what the Security List allows.
The symptom is a connection that hangs rather than refuses, which reads like a
routing problem.

Look at the chain first — you need to insert *before* the REJECT rule, and its
line number varies by image:

```bash
sudo iptables -L INPUT --line-numbers
```

Then insert (adjust `6` to land above the REJECT line):

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
```

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
```

```bash
sudo netfilter-persistent save
```

Without that last command the rules vanish on reboot.

## 6. Install Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
```

```bash
sudo usermod -aG docker $USER
```

Log out and back in for the group change to apply.

## 7. Get the code and the index onto the VM

Create the directories:

```bash
sudo mkdir -p /opt/palate && sudo chown $USER:$USER /opt/palate
```

Clone the repo on the VM:

```bash
git clone YOUR_REPO_URL /opt/palate/app
```

`chroma_db/` is gitignored — 553MB of derived artifact has no business in git —
so it transfers separately. **From your Windows machine**, in Git Bash:

```bash
scp -r "C:/Users/jains/Desktop/Palate_git/restarunt-Rec/chroma_db" ubuntu@YOUR_PUBLIC_IP:/opt/palate/
```

Expect several minutes. Verify the size landed intact before going further — a
truncated transfer produces exactly the silent empty-index failure this whole
setup is built to avoid:

```bash
du -sh /opt/palate/chroma_db
```

It must read **553M**.

## 8. Configure

```bash
cd /opt/palate/app/deploy/oracle && cp .env.example .env
```

Edit `.env`. Copy `mongo_url`, `MONGO_DB` and `RECOMMENDER_TOKEN` verbatim from
the old `env.yaml` — generating a fresh token here means `/index/missing` starts
401ing the moment Vercel points at it.

For `SITE_ADDRESS` with no domain of your own, use **sslip.io**: it resolves
`140-238-1-2.sslip.io` to `140.238.1.2`, and Let's Encrypt will issue for it
because sslip.io is on the Public Suffix List (so its rate limits are per
subdomain, not shared).

**Then allowlist the VM in Atlas.** MongoDB Atlas restricts by source IP and
your address just changed. Atlas → Network Access → Add IP Address → the VM's
public IP. Miss this and every request fails on the Mongo call with a timeout
that says nothing about allowlists.

## 9. Build and start

```bash
cd /opt/palate/app/deploy/oracle && docker compose up -d --build
```

First build takes 10–20 minutes on two ARM cores: torch is a large wheel and
`chroma-hnswlib` may compile from source. Watch it:

```bash
docker compose logs -f recommender
```

## 10. Verify — properly

`GET /health` exists for exactly this step. It reports the vector count in
`places_v2` and returns **503 when the collection is empty or unreadable**,
because Chroma treats a missing `persist_directory` as a new empty collection
rather than an error — so without it a wrong mount gives you a container that
starts cleanly, serves `/docs`, and returns nothing for every query.

Check the container's own view first:

```bash
docker compose ps
```

`(healthy)` means the index mounted and has vectors in it. `(unhealthy)` means
it did not. That is the entire signal, and it is why the healthcheck is worth
having.

For the detail, send the token — `status` and `count` are public, the resolved
path and write-guard state are not:

```bash
curl -s -H "x-recommender-token: YOUR_TOKEN" https://YOUR_SITE_ADDRESS/health
```

```json
{
  "collection": "places_v2",
  "chromaDir": "/data/chroma_db",
  "embedDim": 384,
  "writeGuard": "enforced",
  "status": "ok",
  "count": 15169
}
```

`chromaDir` is the *resolved* path, which is what catches a relative
`CHROMA_DIR` landing against an unexpected working directory. If `count` is 0,
compare it against what is actually on the volume:

```bash
docker compose exec recommender ls -la /data/chroma_db
```

Finally, run a real query — health proves vectors exist, not that ranking works.
Open `https://YOUR_SITE_ADDRESS/docs`, expand `POST /recommend`, hit **Try it
out**, and send a realistic payload. **A 200 with an empty array is a failure,
not a pass.**

If TLS did not provision, Caddy will say why:

```bash
docker compose logs caddy
```

Almost always it is port 80 unreachable, which means step 5.

## 11. Cut Vercel over

Update the recommender base URL in the Vercel project's environment variables to
`https://YOUR_SITE_ADDRESS`, then redeploy. Exercise both callers — the
dashboard's `/nearby` and a group shortlist — before deleting anything.

Now that the host is always-on, the 3-second `/nearby` timeout should never fall
back to distance order again. If it still does, the recommender is genuinely
slow rather than cold, and that is worth investigating separately.

## 12. Decommission GCP

Only after the above passes. In order:

```bash
gcloud run services delete palate-recommender --region asia-south1
```

Then delete the Artifact Registry repo (`cloud-run-source-deploy`) and empty the
`run-sources-*` staging bucket. Those two are where the money actually was, so
skipping them means paying for a service you no longer run.

Leave the **$5 budget alert** in place. The account stays open, and an alert on
a project you have stopped watching is worth more than one you check daily.

## 13. Keep it alive

Two loose ends worth closing:

- **Unattended upgrades** for security patches, since you own the OS now.
- **Idle reclamation** (step 2). If Oracle flags the instance, a small cron that
  exercises the service — a real `/recommend` call every few minutes — both
  raises utilisation and functions as an uptime check.

## Rolling back

Nothing was deleted or edited on the Cloud Run path. The root `Dockerfile`,
`.dockerignore` and `.gcloudignore` are byte-for-byte unchanged, so the original
deploy command still works as documented in the main README. Once you are
confident, delete `Dockerfile.arm64`'s counterpart — the root `Dockerfile`,
`.gcloudignore`, and the Cloud Run section of the main README — and this
directory becomes the only deployment story.
