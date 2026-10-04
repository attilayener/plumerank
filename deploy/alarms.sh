#!/usr/bin/env bash
#
# PlumeRank -- the three CloudWatch alarms the docs describe (drift high, drift low, silence). Run from the OPERATOR's
# laptop, never from the instance: put-metric-alarm needs cloudwatch:PutMetricAlarm,
# and the instance profile (launch.md section 2) deliberately carries only PutMetricData
# into the PlumeRank namespace. The box writes the metrics; the operator owns the alarms.
#
#   AWS_REGION=us-west-2 PR_INSTANCE_ID=i-... bash deploy/alarms.sh       # create or update
#   AWS_REGION=us-west-2 PR_EIP=44.242.91.201 bash deploy/alarms.sh       # id looked up from the EIP
#   ... PR_ALARM_SNS_ARN=arn:aws:sns:... bash deploy/alarms.sh            # with notifications
#
# Every alarm watches the box's own InstanceId dimension (bootstrap.sh writes the id into
# the env file and telemetry.py tags every datapoint with it), so no other publisher --
# a laptop with credentials, a second box -- can keep this box's alarms green.
#
# Idempotent: put-metric-alarm creates or overwrites by name, so a re-run is the update
# (and replaces the dimensionless alarms of the same names an older version created).
#
# What is watched (plumerank/telemetry.py publishes all three metrics):
#
#   PlumeRank-alert-rate-drift-high / -low
#       RealisedAlertRate (alerts per camera-window over the wall's fire-free CONFUSE
#       windows only, UNCAPPED -- api/main.py Replay.night_metrics) outside BUDGET_ALERTS_PER_CAMERA_NIGHT x (1 +/- BUDGET_DRIFT_TOLERANCE)
#       = 2.0 x (1 +/- 0.15) = [1.70, 2.30], hourly average. Missing data is NOT breaching:
#       the replay only runs while someone is watching, so an idle hour has no datapoint
#       and that is the normal overnight state, not drift.
#   PlumeRank-heartbeat-silence
#       Heartbeat, published every HEARTBEAT_SECONDS (60) whether or not anyone watches.
#       Fewer than one datapoint in each of three consecutive 5-minute periods, and missing
#       data IS breaching -- a dead box publishes nothing, and that is the case this exists for.
#
# tests/test_deploy.py asserts the thresholds here match plumerank/constants.py.

set -Eeuo pipefail

REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-west-2}}"
NAMESPACE="PlumeRank"
DRIFT_HIGH="2.3"
DRIFT_LOW="1.7"

INSTANCE_ID="${PR_INSTANCE_ID:-}"
if [ -z "$INSTANCE_ID" ] && [ -n "${PR_EIP:-}" ]; then
  INSTANCE_ID="$(aws ec2 describe-addresses --region "$REGION" --public-ips "$PR_EIP" \
    --query 'Addresses[0].InstanceId' --output text)"
fi
case "$INSTANCE_ID" in
  i-*) ;;
  *) printf '[plumerank-alarms] set PR_INSTANCE_ID=i-... or PR_EIP=<elastic ip>\n' >&2; exit 2 ;;
esac
DIMENSIONS="Name=InstanceId,Value=${INSTANCE_ID}"

actions=()
if [ -n "${PR_ALARM_SNS_ARN:-}" ]; then
  actions=(--alarm-actions "$PR_ALARM_SNS_ARN" --ok-actions "$PR_ALARM_SNS_ARN")
fi

# The silence alarm also REBOOTS the box. 2026-10-04: an out-of-memory wedge left the box
# accepting TCP but answering nothing for two hours while EC2's own status checks stayed
# green, so EC2 auto-recovery would never have fired. The service withholds Heartbeat only
# when it is truly stuck (api/main.py: a wall that failed MAX_TICK_FAILURES ticks in a row,
# or a process that is not running); a wall that is unavailable by design keeps beating.
# CloudWatch runs an action once per transition INTO ALARM, not repeatedly while in it, so
# a box whose telemetry stays silent after a reboot is rebooted once, not in a loop.
reboot_actions=(--alarm-actions "arn:aws:automate:${REGION}:ec2:reboot")
if [ -n "${PR_ALARM_SNS_ARN:-}" ]; then
  reboot_actions=(--alarm-actions "arn:aws:automate:${REGION}:ec2:reboot" "$PR_ALARM_SNS_ARN"
                  --ok-actions "$PR_ALARM_SNS_ARN")
fi

say() { printf '[plumerank-alarms] %s\n' "$*"; }

aws cloudwatch put-metric-alarm --region "$REGION" \
  --alarm-name PlumeRank-alert-rate-drift-high \
  --alarm-description "RealisedAlertRate above 2.0 x 1.15 alerts/camera-night (hourly mean)" \
  --namespace "$NAMESPACE" --metric-name RealisedAlertRate --dimensions "$DIMENSIONS" \
  --statistic Average --period 3600 --evaluation-periods 1 \
  --threshold "$DRIFT_HIGH" --comparison-operator GreaterThanThreshold \
  --treat-missing-data notBreaching ${actions[@]+"${actions[@]}"}
say "PlumeRank-alert-rate-drift-high  > ${DRIFT_HIGH}"

aws cloudwatch put-metric-alarm --region "$REGION" \
  --alarm-name PlumeRank-alert-rate-drift-low \
  --alarm-description "RealisedAlertRate below 2.0 x 0.85 alerts/camera-night (hourly mean)" \
  --namespace "$NAMESPACE" --metric-name RealisedAlertRate --dimensions "$DIMENSIONS" \
  --statistic Average --period 3600 --evaluation-periods 1 \
  --threshold "$DRIFT_LOW" --comparison-operator LessThanThreshold \
  --treat-missing-data notBreaching ${actions[@]+"${actions[@]}"}
say "PlumeRank-alert-rate-drift-low   < ${DRIFT_LOW}"

aws cloudwatch put-metric-alarm --region "$REGION" \
  --alarm-name PlumeRank-heartbeat-silence \
  --alarm-description "No Heartbeat datapoint for 15 minutes: the box or its telemetry is down" \
  --namespace "$NAMESPACE" --metric-name Heartbeat --dimensions "$DIMENSIONS" \
  --statistic SampleCount --period 300 --evaluation-periods 3 \
  --threshold 1 --comparison-operator LessThanThreshold \
  --treat-missing-data breaching "${reboot_actions[@]}"
say "PlumeRank-heartbeat-silence      < 1 sample / 5 min, x3  -> reboots the instance"

say "region ${REGION}, instance ${INSTANCE_ID}. Check: aws cloudwatch describe-alarms --region ${REGION} --alarm-name-prefix PlumeRank"
