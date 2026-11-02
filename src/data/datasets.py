"""Dataset configuration."""

from pathlib import Path

from .types import DatasetSpec


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS: dict[str, DatasetSpec] = {
    "dry_bean": DatasetSpec(
        dataset_name="dry_bean",
        path=PROJECT_ROOT
        / "dataset"
        / "DryBeanDataset"
        / "Dry_Bean_Dataset.csv",
        label_column="Class",
        problem_type="multiclass",
    ),

    "online_shoppers": DatasetSpec(
        dataset_name="online_shoppers",
        path=PROJECT_ROOT
        / "dataset"
        / "online_shop"
        / "online_shoppers_intention.csv",
        label_column="Revenue",
        problem_type="binary",
    ),

    "particle": DatasetSpec(
        dataset_name="particle",
        path=PROJECT_ROOT
        / "dataset"
        / "particle"
        / "MiniBooNE_PID.csv",
        label_column="label",
        problem_type="binary",
    ),

    "jannis": DatasetSpec(
        dataset_name="jannis",
        path=PROJECT_ROOT
        / "dataset"
        / "jannis"
        / "jannis.csv",
        label_column="label",
        problem_type="multiclass",
    ),

    "connect4": DatasetSpec(
        dataset_name="connect4",
        path=PROJECT_ROOT
        / "dataset"
        / "connect4"
        / "connect4.csv",
        label_column="label",
        problem_type="multiclass",
    ),

    "skin": DatasetSpec(
        dataset_name="skin",
        path=PROJECT_ROOT
        / "dataset"
        / "skin"
        / "skin.csv",
        label_column="label",
        problem_type="binary",
    ),

    "cod_rna": DatasetSpec(
        dataset_name="cod_rna",
        path=PROJECT_ROOT
        / "dataset"
        / "cod_rna"
        / "cod_rna.csv",
        label_column="label",
        problem_type="binary",
    ),

    "bng_vote": DatasetSpec(
        dataset_name="bng_vote",
        path=PROJECT_ROOT
        / "dataset"
        / "bng_vote"
        / "bng_vote.csv",
        label_column="label",
        problem_type="binary",
    ),

    "bng_mushroom": DatasetSpec(
        dataset_name="bng_mushroom",
        path=PROJECT_ROOT
        / "dataset"
        / "bng_mushroom"
        / "bng_mushroom.csv",
        label_column="label",
        problem_type="binary",
    ),

    "click": DatasetSpec(
        dataset_name="click",
        path=PROJECT_ROOT
        / "dataset"
        / "click"
        / "click.csv",
        label_column="label",
        problem_type="binary",
    ),

    "road_safety": DatasetSpec(
        dataset_name="road_safety",
        path=PROJECT_ROOT
        / "dataset"
        / "road_safety"
        / "road_safety.csv",
        label_column="label",
        problem_type="binary",
    ),

    "kddcup99": DatasetSpec(
        dataset_name="kddcup99",
        path=PROJECT_ROOT
        / "dataset"
        / "kddcup99"
        / "kddcup99.csv",
        label_column="label",
        problem_type="binary",
    ),
}


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    """ Get the dataset configuration by name. """
    try:
        return DATASETS[dataset_name]
    except KeyError as error:
        available = ", ".join(sorted(DATASETS))
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}; available datasets: {available}"
        ) from error
