# TIC Risk

Command-line tool for running the [TIC Risk Open Model](https://huggingface.co/tic-oss/tic-risk-v2) on
your own machine.

TIC Risk Open is a 232M-parameter model that predicts the risk of bankruptcy for Swedish limited
companies and can also be used to understand how to train a similar model for other countries.

TIC Risk Open was trained on public records for every limited company in Sweden, with no personal
data, and reads each company's history as a sequence of events: annual reports and their figures,
late-filing fees, tax registrations, enforcement debts, payment orders, board changes (without
names) and related companies.

For each company it predicts:

- Bankruptcy within 12 months
- Bankruptcy within 24 months
- New or increased enforcement debt within 12 months
- Tax registration revoked for misconduct within 12 months
- Overdue annual report within 12 months

The approach is not specific to Sweden. It works in any country where credit bureaus or other
organisations have access to company events.

The [paper](paper/tic-risk-v2.pdf) describes how the model was trained and how it compares with TabICL.

## Features

- Runs offline on the CPU. No GPU, database or API needed. The network is only used to download
  the model.
- A single binary for Windows and Linux.
- Terminal report with risk class, 90% prediction interval, monthly risk history and the events
  that drive the score.
- JSON output for scripting.
- A `check` command to validate your own input files.
- Four example companies in [`examples/`](examples), also included in the release downloads.

## Installation

1. Download the binary for your platform from
   [Releases](https://github.com/the-intelligence-company/tic-risk/releases) and verify it against
   the `.sha256` file:
   - Windows: `tic-risk-<version>-x86_64-windows.zip`, signed by The Intelligence Company AB (publ)
   - Linux: `tic-risk-<version>-x86_64-linux.tar.gz`, a static binary for any x86-64 distribution

   Both include the four example companies in `examples/`.
2. Download the model (about 900 MB) into `tic-risk-model`:

   ```
   tic-risk download
   ```

   Every file is checked against its SHA-256 checksum, and an interrupted download continues where
   it stopped. Use `--dir <dir>` to download somewhere else. You can also download the model
   yourself from [Hugging Face](https://huggingface.co/tic-oss/tic-risk-v2).

## Quick start

Windows (cmd or PowerShell):

```
.\tic-risk.exe download
.\tic-risk.exe info
.\tic-risk.exe score --input examples\company-b
.\tic-risk.exe score --input examples\company-a --months-back 60
.\tic-risk.exe score --input examples\company-c --as-of 2025-10-15 --as-of 2026-04-15 --as-of 2026-10-15
.\tic-risk.exe score --input examples\company-d --question bankruptcy_12m --show-input
.\tic-risk.exe check --input examples\company-b
```

Linux:

```
tar -xzf tic-risk-<version>-x86_64-linux.tar.gz
./tic-risk download
./tic-risk info
./tic-risk score --input examples/company-b
./tic-risk check --input examples/company-b
```

The CLI looks for the model in the current directory and in `./tic-risk-model`, then next to the
binary. You can also point to it with `--model <dir>` or the `TIC_RISK_MODEL` environment variable.

`download` uses your proxy settings (`HTTPS_PROXY`, `NO_PROXY`). Set `HF_ENDPOINT` to download from
a Hugging Face mirror.

## Example companies

| Example | Bankruptcy within 12 months | Risk class | Notes |
|---|---|---|---|
| `company-b` | 20.1% | 1 | Payment orders piling up, late fees, annual report deficiencies |
| `company-a` | 7.4% | 1 | Late fees on tax returns, corrected payroll declarations |
| `company-c` | 3.9% | 2 | Findings in annual reports, registry cases |
| `company-d` | 0.9% | 3 | Newly registered company |

The examples are in [`examples/`](examples) and in the release downloads. They are based on real
companies with identifying details removed. IDs are replaced, dates are shifted by a few days,
and amounts and ratios are changed by a few percent, so the results are close to but not the same
as those of the original companies.

## Usage

### score

```
tic-risk score --input <dir> [options]
```

The report shows:

- **Headline:** the probability of bankruptcy within 12 months, the risk class (1 = highest risk,
  5 = lowest), a score from 0 to 100 (100 = lowest risk) and the 90% prediction interval.
- **Risk history:** the 12-month bankruptcy probability for each of the last 36 months, with the
  company's events shown below the chart.
- **All predictions:** each of the five predictions with its interval.
- **Drivers:** how much the 12-month probability changes when each group of events is left out
  (enforcement debt, payroll, annual reports, filings, board and so on). Red raises the risk, green
  lowers it.
- **Over time:** with several `--as-of` dates, a chart and a table of all predictions by date.

The report is colored in a terminal and plain when piped. Set `NO_COLOR` to turn colors off.

| Option | Description |
|---|---|
| `--as-of <date>` | Score as of this date (YYYY-MM-DD). Can be repeated. Defaults to the dates in `meta.json`. |
| `--question <name>` | Only this prediction. Can be repeated. Defaults to all five. |
| `--months-back <n>` | Months of history in the chart (default 36, `0` hides the chart). |
| `--show-input` | Print every event the model reads to stderr. |
| `--explain` | With `--json`: add one line per event group with its effect on the score. |

The prediction names are `bankruptcy_12m`, `bankruptcy_24m`, `enforcement_debt_12m`,
`fskatt_revoked_12m` and `report_overdue_12m`.

### JSON output

`--json` prints one JSON line per date and prediction instead of the report:

```
tic-risk --json score --input examples/company-b
{"company":1,"as_of":"2026-10-15","question":"bankruptcy_12m","horizon_months":12,
 "p_raw":0.842,"p_pit":0.201,"band90":[0.0,0.929],"risk_class":1,"score":0,
 "term_3_6_12_24_raw":[0.175,0.421,0.842,0.971],"tokens":127,"cells":576,"neighbours":0,
 "truncated":false,"unknown_ids":0}
```

| Field | Description |
|---|---|
| `p_pit` | Calibrated probability that the event happens within the horizon |
| `band90` | 90% prediction interval |
| `p_raw` | Uncalibrated model probability |
| `term_3_6_12_24_raw` | Uncalibrated cumulative probability at 3, 6, 12 and 24 months |
| `risk_class`, `score` | Risk class (1–5) and score (0–100), for `bankruptcy_12m` only |
| `tokens`, `neighbours` | Number of events and related companies the model read |
| `truncated` | `true` if the history was too long and the oldest events were dropped |
| `unknown_ids` | Number of codes the model doesn't recognize |

### check

```
tic-risk check --input <dir>
```

Validates an input directory before scoring and reports:

- invalid JSON lines and dates
- unknown event types, attributes and codes
- implausible values
- events known before they happened, or dated after the input's read time
- companies missing from `lives.csv`
- dates the company can't be scored on

Exits with status 1 if the input can't be scored.

### Speed

A prediction takes 1–3 seconds on a desktop, most of which is loading the model. The full report
with the chart and drivers takes a few seconds, or up to about 15 seconds on a laptop. The model
uses all CPU cores.

## Input format

One directory per company:

```
meta.json        {"format":"tic-risk-input/1","focal":1,"read_at":…,"as_of":["2026-10-01"],"pseudonyms":true}
lives.csv        company_id,legal_form,registered,ceased
events.jsonl     {"c":1,"k":"payroll.declared","t":"2025-09-30","ka":"2025-10-07","est":false,"s":"skatteverket","a":{"amount":3603}}
relations.jsonl  {"a":{"t":"company","id":1},"k":"role.auditor","b":{"t":"company","id":2},"from":…,"to":null,"ka":…,"est":false,"x":{…}}
economy.jsonl    national economic series: {"series":…,"period":…,"value":…,"published":…}
sector.jsonl     industry key figures (can be empty)
calendar.jsonl   publication years of the industry figures (can be empty)
```

- `t` is the date of the event and `ka` the date it became known. The model only uses events known
  on or before the as-of date.
- Lines are read in file order.
- IDs are arbitrary numbers.
- `meta.json` can include a `name`, which the report shows instead of the company ID.

The model was trained on Swedish limited companies, so events must use the same codes. The code
lists are in `crates/risk-tokens/texts/`, and codes the model doesn't know are treated as unknown.
The example companies use every part of the format, and `check` shows how the model reads your
files.

## Results

Evaluated on a test set of companies that were excluded from all training, compared with
TabICL on 605 features built from the same data:

| | TIC Risk | TabICL |
|---|---|---|
| Bankruptcy within 12 months, AUC | 0.924 | 0.913 |
| Bankruptcy within 24 months, AUC | 0.890 | 0.853 |
| Share of 12-month bankruptcies in the riskiest 10% | 76% | 73% |

Results for all five predictions, including PR-AUC and calibration, are in the
[model card](https://huggingface.co/tic-oss/tic-risk-v2) and the paper.

## Using it in production

This tool is meant for trying out the model. In production we run the model on a GPU, where a
prediction takes under 10 ms per company.

To score real companies you need their current events in the input format above. You can build
them from your own sources or get them from a credit bureau. TIC also offers an API with company
events that you can combine with your own internal data to run the model privately.

For more examples or help scoring a real company, contact info@tic.io.

## Extending the model

The model is meant as a starting point:

- **Your own data:** train a LoRA adapter on your own companies and outcomes, and compare it with the
  base model on the same companies. Adapter training and loading are not included in this release
  yet.
- **Other countries:** the same approach can be used to train a model for another country from its
  company events. We recommend mapping local codes to a common standard, such as NACE for industry
  codes.
- **Reference implementation:** the weights are a standard safetensors file, and
  `crates/risk-engine/src/cpu.rs` implements the full forward pass.

## Building from source

Requires Rust 1.94 or later:

```
cargo build --release -p tic-risk
```

The binary is written to `target/release/tic-risk` (`tic-risk.exe` on Windows). The Linux release
is built as a static musl binary in a `rust:alpine` container:

```
docker run --rm -v "$PWD":/work -w /work -e RUSTFLAGS="-C target-feature=+crt-static" rust:1.99.1-alpine \
    cargo build --release --target x86_64-unknown-linux-musl -p tic-risk
```

## Project structure

| Crate | Description |
|---|---|
| `tic-risk` | The CLI: report, `check` and JSON output |
| `risk-engine` | CPU inference, input files, calibration and drivers |
| `risk-model` | Vocabulary and sample packing |
| `risk-tokens` | Tokenizer and sampler |
| `risk-eval` | Prediction tasks, labels, dates, and economic and industry series |
| `risk-store` | Record types: companies, annual report figures, labels and source coverage |

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. The model
on Hugging Face uses the same license.
