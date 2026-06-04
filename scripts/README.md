# Numbered Pipeline Scripts

Run order:

1. `00_data_download/00_download_sources.py`: download configured public raw source files.
2. `01_pipeline/01_run_pipeline.sh`: run the Snakemake pipeline end to end.
3. `02_validation/02_validate_outputs.py`: validate generated processed outputs.

Each script has a user-parameter section at the top. Routine path and source
changes should still be made in `configs/config.yaml`.
