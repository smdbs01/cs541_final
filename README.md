# Introduction

TO BE WRITTEN

# Installation

The project uses [*uv*](https://docs.astral.sh/uv/) for package management.

Make sure you have *uv* installed, if not, follow [the documentation](https://docs.astral.sh/uv/getting-started/installation/).

# Prepare dataset

```bash
uv run download_data.py
```

# Preprocess Pose Data

```bash
uv run preprocess.py
```

# Train 

```bash
uv run train.py 
```

# Test

```bash
uv run test.py
```