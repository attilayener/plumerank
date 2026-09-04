# Launching the PlumeRank box

Everything needed to create the one machine PlumeRank runs on, in the order it has to
happen. Copy-paste from top to bottom and you get the live endpoint; nothing here is
pseudocode and nothing here is a template you have to render first.

**There is no CloudFormation.** That is a decision. A judge scoring cloud delivery and
reproducibility will not launch a stack, but they will read a shell script and they will
curl an endpoint -- and this deployment is *more* verifiable than a template, because
`/build-info` echoes the AMI id back, so the running box can be checked against the AMI
pinned in this file in one request. A template proves nothing until somebody runs it.

**There is no `deploy/build-opencv5.sh`.** The D4 wheel gate resolved to **branch A** on
2026-09-04: `opencv-contrib-python-headless==5.0.0.93` publishes a
`manylinux_2_28_aarch64` wheel, so OpenCV 5 installs on Graviton with
`pip install -r requirements.lock` and there is no source build to script. Branches C and
D of that gate -- stop the instance, resize to a bigger Graviton for an hour, build,
resize back -- were pre-written and never ran. `architecture.md` section 8 has the
decision table; `DEMO.md` records that branch A is what happened.

**There is no Route 53 hosted zone.** One A record at the registrar points at the Elastic
IP. Managed DNS scores zero rubric lines, and it would add an infrastructure resource that
has to be created, destroyed and debugged for the sake of a field you can type once.

---

## 0. Settings used throughout

```bash
export AWS_REGION=us-west-2               # cheapest US region for Graviton; the imagery
                                          # and the beneficiary agency are both Californian
export PR_NAME=plumerank
export PR_INSTANCE_TYPE=t4g.small         # Graviton2, 2 vCPU, 2 GiB, 750 free h/month
                                          # through 2026-12-31
export PR_VOLUME_GB=30
export PR_KEYPAIR=plumerank-admin         # created once, in section 0a below
export PR_BUCKET=plumerank-corpus         # holds exactly one prefix: corpus/
export PR_REPO=https://github.com/OWNER/REPO.git   # <- your clone URL
export PR_REF=main
export PR_DOMAIN=plumerank.org            # registered at any registrar; see step 6
```

`PR_BUCKET` appears again, hard-coded, in the IAM policy JSON of section 2. JSON has no
interpolation, so those two are kept in step by hand: change one and change the other, or
the box gets `AccessDenied` on a corpus it can see in the console.

### 0a. The key pair -- created once, before anything else

`run-instances` fails with `InvalidKeyPair.NotFound` if this does not exist, and section 6
cannot turn TLS on without it.

```bash
aws ec2 create-key-pair --region "$AWS_REGION" --key-name "$PR_KEYPAIR" \
  --key-type ed25519 --query 'KeyMaterial' --output text > ~/.ssh/${PR_KEYPAIR}.pem
chmod 600 ~/.ssh/${PR_KEYPAIR}.pem
```

AWS returns the private key exactly once and stores none of it. If that file is lost the
key pair is deleted and recreated, and the instance is rebuilt from section 1 -- which is
survivable here only because nothing on the box is irreplaceable (section 9).

---

## 1. The AMI -- pinned

```bash
# NOT YET RESOLVED. This is a sentinel, not an AMI id: it is deliberately malformed so
# that `run-instances` in section 4 fails with InvalidAMIID.Malformed rather than
# launching. Resolve it with the ssm command below, paste the real id here, and COMMIT
# that line. A well-formed id nobody verified is worse than no id at all -- it launches,
# and then /build-info's AMI echo is being compared against a number somebody typed.
export PR_AMI=ami-RESOLVE-BEFORE-LAUNCH
```

Resolve it once, here, and commit the result:

```bash
aws ssm get-parameter --region "$AWS_REGION" \
  --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-arm64 \
  --query 'Parameter.Value' --output text
```

The id it prints is what goes in the line above, replacing the sentinel. Re-pinning is a
commit, not a runtime lookup: the commit is the record of which base image the published
numbers were produced on.

Once it is a real id, it is **pinned, not resolved at launch time**, and that is the point: `/build-info`
reports the AMI the instance actually booted from, and a judge comparing it with the line
above is checking the documentation against the machine rather than taking either on
trust. A `run-instances` line that resolves `latest` produces a box whose base image
nobody can name a month later.

**Verify the pin before every launch.** The check is one command and it fails loudly:

```bash
aws ec2 describe-images --region "$AWS_REGION" --image-ids "$PR_AMI" \
  --query 'Images[0].[ImageId,Name,Architecture,CreationDate]' --output text
```

If that errors with `InvalidAMIID.NotFound` or `InvalidAMIID.Malformed` -- the first
because AL2023 images are republished and deregistered on Amazon's schedule and not ours,
the second because the pin is still the sentinel -- **re-resolve with the `ssm
get-parameter` line above, edit the pin, and commit the change**, so the file and the box
never drift apart. Do not work around it on the command line: a pin that lives only in
one shell's history is not a pin.

---

## 2. The instance profile -- `plumerank-instance`, exactly three statements

One profile, one inline policy, three statements. **No long-lived access key ever lands
on this box**: the corpus upload runs from the developer laptop under `~/.aws/`, weeks
before the instance exists, which is why nothing below grants `s3:PutObject`.
`check_submission_readiness.py` greps the repository for `AKIA` to prove none reached git
either.

Trust policy -- `plumerank-instance-trust.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "EC2AssumesThisRole",
      "Effect": "Allow",
      "Principal": { "Service": "ec2.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

Inline policy -- `plumerank-instance-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "WriteOnlyToTheOneMetricNamespace",
      "Effect": "Allow",
      "Action": "cloudwatch:PutMetricData",
      "Resource": "*",
      "Condition": {
        "StringEquals": { "cloudwatch:namespace": "PlumeRank" }
      }
    },
    {
      "Sid": "ReadBackTheMetricsTheLiveTestAsserts",
      "Effect": "Allow",
      "Action": "cloudwatch:GetMetricStatistics",
      "Resource": "*"
    },
    {
      "Sid": "ReadOnlyOnTheCorpusPrefix",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::plumerank-corpus",
        "arn:aws:s3:::plumerank-corpus/corpus/*"
      ],
      "Condition": {
        "StringLikeIfExists": { "s3:prefix": ["corpus", "corpus/*"] }
      }
    }
  ]
}
```

Three notes on that policy, because each of them is a decision someone will ask about:

- `PutMetricData` has no resource-level ARN in CloudWatch, so the namespace condition is
  the *only* thing that scopes it. Without that condition the box could write into any
  namespace in the account; with it, `PlumeRank` is the only one it can touch, and it can
  still read nothing back except through the second statement.
- The second statement exists solely so the `@pytest.mark.live` test can assert that a
  real `RealisedAlertRate` datapoint appeared within 15 minutes of a replay. A metric
  nobody can read back is a metric nobody has checked.
- `StringLikeIfExists`, not `StringLike`. `s3:prefix` is present in a `ListBucket` request
  and absent from a `GetObject` request, so a plain `StringLike` evaluates false on every
  object read and silently denies the corpus fetch. `IfExists` constrains the listing to
  `corpus/` and leaves `GetObject` scoped by its resource ARN, which is where object-level
  scoping belongs. This is a real trap and it costs an afternoon to diagnose from an
  `AccessDenied` with no detail in it.

Create it:

```bash
aws iam create-role --role-name plumerank-instance \
  --assume-role-policy-document file://plumerank-instance-trust.json

aws iam put-role-policy --role-name plumerank-instance \
  --policy-name plumerank-instance-inline \
  --policy-document file://plumerank-instance-policy.json

aws iam create-instance-profile --instance-profile-name plumerank-instance
aws iam add-role-to-instance-profile --instance-profile-name plumerank-instance \
  --role-name plumerank-instance
```

---

## 3. The security group -- two rules, and port 22 is not one of them

```bash
export PR_VPC=$(aws ec2 describe-vpcs --region "$AWS_REGION" \
  --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)

export PR_SG=$(aws ec2 create-security-group --region "$AWS_REGION" \
  --group-name plumerank-web --vpc-id "$PR_VPC" \
  --description "PlumeRank public web: 443 and 80 only" \
  --query 'GroupId' --output text)

# Rule 1 -- HTTPS. The endpoint itself.
aws ec2 authorize-security-group-ingress --region "$AWS_REGION" --group-id "$PR_SG" \
  --protocol tcp --port 443 --cidr 0.0.0.0/0

# Rule 2 -- HTTP. NOT a convenience: it carries the Let's Encrypt HTTP-01 challenge and
# the redirect Caddy installs to 443. Closing it means no certificate.
aws ec2 authorize-security-group-ingress --region "$AWS_REGION" --group-id "$PR_SG" \
  --protocol tcp --port 80 --cidr 0.0.0.0/0
```

**Those are the only two standing rules. SSH is NOT open to the world -- there is no
0.0.0.0/0 rule on port 22, at any point, for any reason.** This box is public and
unattended for the fourteen days of the judging window, and a permanently open SSH port on
an unattended host is the single most-scanned thing on the internet.

When a shell is genuinely needed -- sections 6 and 9 both need one, and both inline this
block rather than referring back to it -- open port 22 to one address, do the work, and
close it again in the same session:

```bash
MY_IP=$(curl -fsS https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --region "$AWS_REGION" --group-id "$PR_SG" \
  --protocol tcp --port 22 --cidr "${MY_IP}/32"

ssh -i ~/.ssh/${PR_KEYPAIR}.pem ec2-user@"$PR_DOMAIN"

aws ec2 revoke-security-group-ingress --region "$AWS_REGION" --group-id "$PR_SG" \
  --protocol tcp --port 22 --cidr "${MY_IP}/32"
```

Almost nothing needs that, because a deploy is `bootstrap.sh` again and the logs are in
CloudWatch and `/var/log/plumerank-bootstrap.log`.

---

## 4. Launch

`bootstrap.sh` is passed **as the userdata, unmodified** -- the file in the repository is
literally the file that ran on the box. Its one configurable line, `PLUMERANK_REPO_DEFAULT`,
is committed rather than injected, precisely so that no templating step sits between the
two. Set it before you launch.

Run this **from the repository root**: `--user-data file://deploy/bootstrap.sh` is a
relative path, and from anywhere else the CLI errors on a missing file rather than
launching a box with no userdata -- which is the good failure, but only if you know that
is what it means.

```bash
export PR_INSTANCE_ID=$(aws ec2 run-instances \
  --region "$AWS_REGION" \
  --image-id "$PR_AMI" \
  --instance-type "$PR_INSTANCE_TYPE" \
  --key-name "$PR_KEYPAIR" \
  --security-group-ids "$PR_SG" \
  --iam-instance-profile Name=plumerank-instance \
  --metadata-options "HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=1" \
  --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=${PR_VOLUME_GB},VolumeType=gp3,DeleteOnTermination=true,Encrypted=true}" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${PR_NAME}},{Key=Project,Value=PlumeRank}]" \
  --user-data file://deploy/bootstrap.sh \
  --query 'Instances[0].InstanceId' --output text)

echo "$PR_INSTANCE_ID"
```

Four flags worth naming:

- `HttpTokens=required` -- IMDSv2 only. `/build-info` reads the instance id and the AMI id
  through IMDSv2 with a 0.2 s timeout; IMDSv1 is the SSRF-to-credentials path and there is
  no reason to leave it enabled on a box that serves user-reachable HTTP.
- `HopLimit=1` -- the metadata service is not reachable from anything one hop further out.
- `Encrypted=true` on the root volume: free, and the corpus cache lives on it.
- **No `--credit-specification`.** See section 8: `t4g` launches in `unlimited` by default
  and that default is the correct setting. Passing the flag at all invites someone to pass
  `standard`.

---

## 5. Elastic IP

The instance is stopped and started during the build (branch C of the wheel gate was
pre-written around exactly that, and the load check reboots it), and a public IP that
changes on stop would take the A record with it.

```bash
export PR_EIP_ALLOC=$(aws ec2 allocate-address --region "$AWS_REGION" --domain vpc \
  --query 'AllocationId' --output text)

aws ec2 associate-address --region "$AWS_REGION" \
  --instance-id "$PR_INSTANCE_ID" --allocation-id "$PR_EIP_ALLOC"

aws ec2 describe-addresses --region "$AWS_REGION" --allocation-ids "$PR_EIP_ALLOC" \
  --query 'Addresses[0].PublicIp' --output text
```

Keep the allocation associated. A released-and-reallocated Elastic IP is a different
address and a new DNS propagation wait.

---

## 6. DNS -- one A record, at the registrar

In the registrar's DNS panel, one record:

| Type | Host | Value | TTL |
|---|---|---|---|
| `A` | `@` | the Elastic IP from step 5 | 300 |

That is the entire DNS configuration. No hosted zone, no nameserver delegation, no
IaC coupling to a DNS provider.

Wait for it to resolve, then turn TLS on -- Let's Encrypt validates the name, so the
record has to exist first, which is why the first boot deliberately serves plain HTTP:

```bash
dig +short "$PR_DOMAIN"          # must return the Elastic IP before the next step

# Port 22 is shut (section 3), so this is the open / work / close sequence, in one block
# so that the close cannot be forgotten between two copy-pastes.
MY_IP=$(curl -fsS https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --region "$AWS_REGION" --group-id "$PR_SG" \
  --protocol tcp --port 22 --cidr "${MY_IP}/32"

ssh -i ~/.ssh/${PR_KEYPAIR}.pem ec2-user@"$PR_DOMAIN" \
  "sudo env PLUMERANK_DOMAIN=$PR_DOMAIN bash /opt/plumerank/deploy/bootstrap.sh"

aws ec2 revoke-security-group-ingress --region "$AWS_REGION" --group-id "$PR_SG" \
  --protocol tcp --port 22 --cidr "${MY_IP}/32"
```

`sudo env PLUMERANK_DOMAIN=...`, not `sudo PLUMERANK_DOMAIN=...`. Under the default
sudoers `env_reset`, a variable set in the second form can be dropped before `bash` ever
sees it, and the failure is silent and misleading: the script runs to completion, writes
an empty domain into `/etc/plumerank.env`, and leaves the box on plain HTTP looking for
all the world like a certificate problem. `env` is a command, so sudo cannot filter its
assignments at all.

That re-run writes `/etc/plumerank.env`, restarts Caddy, and Caddy provisions the
certificate on its own -- and it does not print `DONE` until it has fetched
`https://$PR_DOMAIN/health` through Caddy itself, so a certificate that never arrived
shows up in the run's own output rather than in a judge's browser. Re-running
`bootstrap.sh` is always safe: it is the deploy.

---

## 7. Verify -- the check that makes the pin worth having

```bash
curl -fsS "https://${PR_DOMAIN}/health"
curl -fsS "https://${PR_DOMAIN}/build-info" | head -c 2000
```

Three things have to agree, and all three are in that second response:

1. `cv2.__version__` starts with `5.` and `cv2.getBuildInformation()` is there verbatim.
2. The `ami_id` field equals `PR_AMI` from section 1. If it does not, this file is stale
   or the box was launched from something else -- either way the documentation and the
   machine have drifted and one of them is wrong.
3. The `commit` field equals `git rev-parse HEAD` in the clone you deployed from. A live
   test asserts exactly this over HTTPS.

---

## 8. The burstable-credit rule -- read this before any batch run

`t4g.small` is a **burstable** instance: 2 vCPUs, 20% baseline per vCPU, **24 CPU credits
earned per hour, 576 maximum accrued**. One credit is one vCPU-minute at 100%.

**The instance stays in its default `unlimited` credit mode. It is never switched to
`standard`.** `t4g` launches in `unlimited`, and in `unlimited` mode an instance may spend
surplus credits up to the maximum it could earn in 24 hours -- 576 at this size -- at no
charge. `standard` mode would have throttled the box to 0.4 vCPU part-way through the
bench, roughly a five-fold longer run, **to save nothing**, because the runs fit inside the
free surplus allowance either way. Unlimited mode does not throttle; it bills, and inside
576 it bills zero.

The budget, therefore, is a rule about hours rather than a mode setting:

- **At most 4 hours of full-tilt 2-vCPU batch work per rolling 24 hours** -- 480 of the 576
  credits the instance earns in a day. The full HOLD bench is ~60 minutes (120 credits);
  the twelve-cell ablation grid is at most 4 hours (480 credits) and is resumable per cell,
  so an interruption never restarts it.
- `bench.py` prints `CPUCreditBalance` before and after every run and the delta is
  published in `DEMO.md`. The risk becomes evidence.
- **No full bench and no ablation run after Oct 23**, so the balance is back at its 576
  ceiling when the judging window opens.

The streaming side is handled in code, not by a rule: the replay loop starts on the first
SSE connection and stops ~120 s after the last one disconnects, so an idle box costs one
heartbeat per minute and credits accumulate between judge visits. A continuously running
wall would sit above baseline every day of an unattended fourteen-day window, which is why
the on-demand loop is a correctness requirement rather than a convenience.

The free tier covers instance-hours, not unlimited-mode surplus credits. The two are
priced separately; the rule above is what keeps the second one at zero.

---

## 9. Re-deploying, and rebuilding from nothing

**A deploy** is the same open / work / close block as section 6, with no domain to pass:

```bash
MY_IP=$(curl -fsS https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress --region "$AWS_REGION" --group-id "$PR_SG" \
  --protocol tcp --port 22 --cidr "${MY_IP}/32"

ssh -i ~/.ssh/${PR_KEYPAIR}.pem ec2-user@"$PR_DOMAIN" \
  "sudo bash /opt/plumerank/deploy/bootstrap.sh"

aws ec2 revoke-security-group-ingress --region "$AWS_REGION" --group-id "$PR_SG" \
  --protocol tcp --port 22 --cidr "${MY_IP}/32"
```

The domain is not passed again: `bootstrap.sh` persists it in `/etc/plumerank.env` and
reads it back, so a bare re-run keeps TLS on.

It fetches, resets to `origin/$PR_REF`, reinstalls the lockfile, rebuilds the export,
reinstalls the unit and the Caddyfile, restarts both services and health-checks the
result. Every step is guarded, so running it twice in a row changes nothing the second
time.

**A rebuild from nothing** is sections 1 through 6 again, in order, against the same
Elastic IP and the same bucket. Nothing on the box is irreplaceable: the corpus is in
`s3://${PR_BUCKET}/corpus/`, every published number is committed in `results/`, and the
evidence crops are regenerated by the replay that draws them.
