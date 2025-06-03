# Cluster Manager Utilities

This directory contains utility scripts for debugging, testing, and maintaining the cluster manager.

## Scripts

### `check_cluster_data.py`

Queries the database to show current cluster state including:

- Active nodes and their status
- Latest system resource metrics
- Cluster health summary

Usage: `python utils/check_cluster_data.py`

### `debug_cluster.py`

Comprehensive debugging script that tests all cluster manager components:

- Database initialization
- Node registration
- Monitoring loop components
- System resource collection
- Complete monitoring iteration

Usage: `python utils/debug_cluster.py`

### `validate_cluster.py`

Validation script for testing cluster functionality and AMD GPU integration.
Originally designed for testing cluster validation without full infrastructure.

Usage: `python utils/validate_cluster.py`

## Running Utilities

All utilities should be run from the cluster-manager root directory:

```bash
cd /path/to/cluster-manager
python utils/script_name.py
```

The scripts automatically load configuration from `config/cluster-manager-db.env`.
