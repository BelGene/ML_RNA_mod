from __future__ import annotations

from pathlib import Path

import typer

from rnmod.dataset.build_master_table import build as build_master_table
from rnmod.ingest import ecocyc, go, legacy_pilot, manual_literature, modomics, rhea, uniprot

app = typer.Typer(help="Build the ML_RNA_mod curated RNA-modification protein dataset.")


@app.command("ingest-uniprot")
def ingest_uniprot(
    config: Path = typer.Option(Path("configs/config.yaml"), help="Project config YAML."),
    output: Path = typer.Option(Path("data/interim/uniprot/uniprot_records.parquet"), help="Output parquet."),
) -> None:
    """Normalize cached or fetched UniProt records."""
    frame = uniprot.ingest(config, output)
    typer.echo(f"wrote {len(frame)} UniProt records to {output}")


@app.command("ingest-rhea")
def ingest_rhea(
    config: Path = typer.Option(Path("configs/config.yaml"), help="Project config YAML."),
    output: Path = typer.Option(Path("data/interim/rhea/rhea_reactions.parquet"), help="Output parquet."),
) -> None:
    """Normalize a cached/manual Rhea reaction table."""
    frame = rhea.ingest(config, output)
    typer.echo(f"wrote {len(frame)} Rhea rows to {output}")


@app.command("ingest-go")
def ingest_go(
    config: Path = typer.Option(Path("configs/config.yaml"), help="Project config YAML."),
    output: Path = typer.Option(Path("data/interim/go/go_terms.parquet"), help="Output parquet."),
) -> None:
    """Normalize GO RNA-modification terms from an OBO file."""
    frame = go.ingest(config, output)
    typer.echo(f"wrote {len(frame)} GO rows to {output}")


@app.command("ingest-modomics")
def ingest_modomics(
    config: Path = typer.Option(Path("configs/config.yaml"), help="Project config YAML."),
    output: Path = typer.Option(Path("data/interim/modomics/modomics_records.parquet"), help="Output parquet."),
) -> None:
    """Normalize the permitted manual/API MODOMICS import table."""
    frame = modomics.ingest(config, output)
    typer.echo(f"wrote {len(frame)} MODOMICS records to {output}")


@app.command("ingest-ecocyc")
def ingest_ecocyc(
    config: Path = typer.Option(Path("configs/config.yaml"), help="Project config YAML."),
    output: Path = typer.Option(Path("data/interim/ecocyc/ecocyc_records.parquet"), help="Output parquet."),
) -> None:
    """Normalize the manual EcoCyc/BioCyc import table."""
    frame = ecocyc.ingest(config, output)
    typer.echo(f"wrote {len(frame)} EcoCyc records to {output}")


@app.command("ingest-manual-literature")
def ingest_manual_literature(
    config: Path = typer.Option(Path("configs/config.yaml"), help="Project config YAML."),
    output: Path = typer.Option(Path("data/interim/manual_literature/manual_literature_records.parquet"), help="Output parquet."),
) -> None:
    """Normalize manually curated literature seed records."""
    frame = manual_literature.ingest(config, output)
    typer.echo(f"wrote {len(frame)} manual literature records to {output}")


@app.command("ingest-legacy-pilot")
def ingest_legacy_pilot(
    config: Path = typer.Option(Path("configs/config.yaml"), help="Project config YAML."),
    output: Path = typer.Option(Path("data/interim/legacy_pilot/legacy_pilot_records.parquet"), help="Output parquet."),
) -> None:
    """Normalize the previous EDL933 pilot seed library as a legacy source."""
    frame = legacy_pilot.ingest(config, output)
    typer.echo(f"wrote {len(frame)} legacy pilot records to {output}")


@app.command("build-master")
def build_master(
    config: Path = typer.Option(Path("configs/config.yaml"), help="Project config YAML."),
) -> None:
    """Build master dataset, FASTA, label matrix, manifest, and dataset card."""
    outputs = build_master_table(config)
    for name, path in outputs.items():
        typer.echo(f"{name}: {path}")


if __name__ == "__main__":
    app()

