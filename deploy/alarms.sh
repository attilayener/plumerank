#!/usr/bin/env bash
#
# PlumeRank -- the three CloudWatch alarms the docs describe (drift high, drift low, silence). Run from the OPERATOR's
# laptop, never from the instance: put-metric-alarm needs cloudwatch:PutMetricAlarm,
# and the instance profile (launch.md section 2) deliberately carries only PutMetricData
# into the PlumeRank namespace. The box writes the metrics; the operator owns the alarms.
#
#   AWS_REGION=us-west-2 bash deploy/alarms.sh                 # create or update
#   AWS_REGION=us-west-2 PR_ALARM_SNS_ARN=arn:aws:sns:... bash deploy/alarms.sh
#
# Idempotent: put-metric-alarm creates or overwrites by name, so a re-run is the update.
#
# What is watched (plumerank/telemetry.py publishes all three metrics):
#
#   PlumeRank-alert-rate-drift-high / -low
#       RealisedAlertRate (alerts per camera-night, UNCAPPED -- api/main.py
#       _publish_telemetry) outside BUDGET_ALERTS_PER_CAMERA_NIGHT x (1 +/- BUDGET_DRIFT_TOLERANCE)
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

actions=()
if [ -n "${PR_ALARM_SNS_ARN:-}" ]; then
  actions=(--alarm-actions "$PR_ALARM_SNS_ARN" --ok-actions "$PR_ALARM_SNS_ARN")
fi

say() { printf '[plumerank-alarms] %s\n' "$*"; }

aws cloudwatch put-metric-alarm --region "$REGION" \
  --alarm-name PlumeRank-alert-rate-drift-high \
  --alarm-description "RealisedAlertRate above 2.0 x 1.15 alerts/camera-night (hourly mean)" \
  --namespace "$NAMESPACE" --metric-name RealisedAlertRate \
  --statistic Average --period 3600 --evaluation-periods 1 \
  --threshold "$DRIFT_HIGH" --comparison-operator GreaterThanThreshold \
  --treat-missing-data notBreaching ${actions[@]+"${actions[@]}"}
say "PlumeRank-alert-rate-drift-high  > ${DRIFT_HIGH}"

aws cloudwatch put-metric-alarm --region "$REGION" \
  --alarm-name PlumeRank-alert-rate-drift-low \
  --alarm-description "RealisedAlertRate below 2.0 x 0.85 alerts/camera-night (hourly mean)" \
  --namespace "$NAMESPACE" --metric-name RealisedAlertRate \
  --statistic Average --period 3600 --evaluation-periods 1 \
  --threshold "$DRIFT_LOW" --comparison-operator LessThanThreshold \
  --treat-missing-data notBreaching ${actions[@]+"${actions[@]}"}
say "PlumeRank-alert-rate-drift-low   < ${DRIFT_LOW}"

aws cloudwatch put-metric-alarm --region "$REGION" \
  --alarm-name PlumeRank-heartbeat-silence \
  --alarm-description "No Heartbeat datapoint for 15 minutes: the box or its telemetry is down" \
  --namespace "$NAMESPACE" --metric-name Heartbeat \
  --statistic SampleCount --period 300 --evaluation-periods 3 \
  --threshold 1 --comparison-operator LessThanThreshold \
  --treat-missing-data breaching ${actions[@]+"${actions[@]}"}
say "PlumeRank-heartbeat-silence      < 1 sample / 5 min, x3"

say "region ${REGION}. Check: aws cloudwatch describe-alarms --region ${REGION} --alarm-name-prefix PlumeRank"
