# System Specifications Checker

**Author**: Mauro Risonho de Paula Assumpção <mauro.risonho@gmail.com>  
**License**: MIT License

The `system_specs_checker.py` module inspects the host system and produces a
comprehensive JSON report summarising CPU, GPU, memory, storage, and Python
environment details. It is intended to validate whether a machine is suitable
for deep-learning experimentation.

## Features

- CPU and memory inventory with utilisation metrics
- GPU discovery (NVIDIA, AMD, Metal, or integrated graphics)
- CUDA and PyTorch availability checks
- Storage utilisation, temperature readings, and network information
- Optional JSON export for sharing benchmark results

## Usage

Run the checker directly:

```bash
python system_specs_checker.py --output system_specs.json
```

The command prints a condensed summary to stdout and writes the detailed report
to `system_specs.json`. Omitting `--output` prints the report to stdout only.

### Example: Quick ML Readiness Report

The companion file `system_checker_examples.py` provides reusable snippets. To
run a quick readiness check:

```bash
python system_checker_examples.py
```

## Configuration

The checker relies on common Linux utilities (`lscpu`, `nvidia-smi`, `sensors`,
`lsblk`) when available. If a command is missing the corresponding section of
the report is gracefully skipped.

## Troubleshooting

| Issue | Resolution |
| ----- | ---------- |
| Permission errors when reading `/sys` | Run the script with elevated privileges (e.g. `sudo`) |
| Missing GPU information | Ensure the vendor CLI (`nvidia-smi`, `rocm-smi`) is installed |
| Incorrect Python version | Activate the project virtual environment prior to running |

Refer to `system_checker_examples.py` for additional patterns such as monitoring
temperatures or evaluating storage health.
