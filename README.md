# Network Operations Intelligence Copilot

Anomaly detection and incident triage for a simulated BGP/OSPF router fleet:
Spark feature engineering → SageMaker model tracked in MLflow → LangGraph agent
that writes Bedrock-generated, Pydantic-validated incident reports → FastAPI on
Kubernetes, with an A/B evaluation of alerting strategies and CI/CD.

| Phase | Component | Status |
|---|---|---|
| 1 | Telemetry generator + PySpark rolling-window features | **Done, tested** |
| 2 | Terraform: S3, least-privilege IAM, serverless Spark on AWS Glue | **Done, deployed** (SageMaker resources come with Phase 3) |
| 3 | SageMaker training + endpoint, MLflow tracking | Not started |
| 4 | LangGraph + Bedrock + Pydantic retry loop | Not started |
| 5 | FastAPI, Docker, Kubernetes | Not started |
| 6 | A/B test: threshold vs ML alerting | Not started |
| 7 | GitHub Actions CI/CD | Not started |

## Phase 1: data + Spark

```
data/generate_telemetry.py        simulated fleet telemetry with labeled anomaly episodes
spark_jobs/feature_engineering.py per-router rolling features, date-partitioned Parquet
tests/                            14 tests; Spark tests check window logic by hand-computed values
```

**Telemetry.** 200 routers × 7 days × 5-minute samples = 403,200 rows of
`latency_ms, packet_loss_pct, route_flap_count, bgp_session_state, traffic_mbps`.
Baselines vary by router role (core/aggregation/edge) and follow a diurnal load
curve. About 1% of rows sit inside injected anomaly *episodes* (3–18 consecutive
samples), labeled `latency_spike`, `packet_loss_burst`, `route_flap_storm` or
`session_down`.

**Why per-router features.** A latency spike on a core router (~5 ms → 25 ms) is
still below an edge router's *normal* latency (~22 ms). A global threshold cannot
catch it, but a z-score against the router's own recent history can.

**Why Spark.** Routers are independent, so the work partitions by `router_id` and
runs in parallel. 400K rows would fit in pandas; the justification is production
scale (thousands of routers, sub-minute intervals, weeks of retention = 10⁸+ rows).
The same code runs on EMR/Glue by changing the master and input path.

**Design decisions** (explained in the job's docstring):
- Time-based windows (`rangeBetween` on epoch seconds), not row counts, so missing samples don't silently stretch the window.
- The baseline window excludes the current row, so a spike can't inflate its own mean and std.
- Z-scores are null, not 0, when history is short or variance is zero.
- Output is partitioned by `date`, not `router_id`, to avoid the small-files problem and match time-range reads.

**Validated result** (full dataset, mean |z-score| by row type):

| type | latency \|z\| | loss \|z\| | traffic \|z\| | flaps (15m) | session down |
|---|---|---|---|---|---|
| normal | 0.91 | 0.92 | 1.09 | 0.2 | 0.00 |
| latency_spike | **10.36** | 0.93 | 1.06 | 0.1 | 0.00 |
| packet_loss_burst | 1.87 | **41.61** | 1.08 | 0.2 | 0.00 |
| route_flap_storm | 3.08 | 10.05 | 1.14 | **95.7** | 0.77 |
| session_down | 0.96 | 0.93 | 2.82 | 1.2 | **1.00** |

**Known limitation:** during a long `session_down`, the collapsed traffic enters the
trailing baseline, so the traffic z-score fades as the episode goes on (|z| only 2.8).
Phase 3 should use `bgp_session_down` and `traffic_pct_change` for this case, or a
baseline that freezes while an anomaly is active.

**Bugs hit and fixed:**
- pandas writes nanosecond Parquet timestamps; Spark 3.5 only reads microseconds (`Illegal Parquet type: INT64 (TIMESTAMP(NANOS,true))`). Fixed in the producer with `coerce_timestamps="us"`.
- PySpark 3.5 imports `distutils`, which Python 3.12 removed. Fixed by adding `setuptools` to the requirements.

## Phase 2: Terraform + serverless Spark on AWS Glue

```
infra/versions.tf   provider, default tags (Project / ManagedBy on every resource)
infra/s3.tf         private encrypted bucket; prefixes raw/ scripts/ features/; script deployed by etag
infra/iam.tf        Glue job role: trust policy + least-privilege permission policy
infra/glue.tf       Glue 5.0 (Spark 3.5) job definition, 2 x G.1X, 20-min timeout, no retries
spark_jobs/run_glue_job.py   upload data, start run, poll, print driver output from CloudWatch
```

The same `feature_engineering.py` runs locally and on Glue; only the `--input` and
`--output` paths change. **Why Glue rather than EMR Serverless:** both are serverless
Spark (no cluster to keep alive; you pay per second for the run). This account is on the
AWS Free plan, which blocks EMR, and Glue runs plain PySpark scripts unchanged.

**IAM design.** A role has two policies:
- The **trust policy** says who can assume the role: only `glue.amazonaws.com`, and only for this account (`aws:SourceAccount`, which guards against the confused-deputy problem).
- The **permission policy** says what the role can do: read `raw/` and `scripts/`, read and write `features/`, and write only to `/aws-glue/*` log groups.

The AWS-managed `AWSGlueServiceRole` is deliberately not used; it grants far more than this job needs. The limits are verified with the IAM policy simulator:

| Action | Resource | Decision |
|---|---|---|
| GetObject | `raw/telemetry.parquet` | allowed |
| PutObject | `features/*` | allowed |
| PutObject / DeleteObject | `raw/telemetry.parquet` | **denied**: the job can't damage its own input |
| PutObject | `scripts/feature_engineering.py` | **denied**: the job can't modify its own code |
| GetObject | any other bucket | **denied** |
| DeleteBucket | this bucket | **denied** |

**First run:** it succeeded in 95 s using 190 DPU-seconds (≈ $0.02) and wrote 403,200 rows into 7 `date=` partitions.
The z-score table from Glue matches the local run exactly.

```bash
terraform -chdir=infra init
terraform -chdir=infra plan -out=tfplan      # review before creating anything
terraform -chdir=infra apply tfplan
python spark_jobs/run_glue_job.py
terraform -chdir=infra destroy               # tear everything down when finished
```

### Run it locally

Spark writes Parquet through Hadoop's filesystem layer, which on Windows needs
`winutils.exe` (only distributed unofficially). On Windows, run the full job on Glue
(above) or in WSL/Linux. The tests run on Windows because they do no file I/O.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python data/generate_telemetry.py
python spark_jobs/feature_engineering.py
pytest -q
```

Requires Java 8, 11 or 17 on `PATH` (`sudo apt install openjdk-17-jre-headless` on Ubuntu).
