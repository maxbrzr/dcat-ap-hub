from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from dcat_ap_hub import BaseProcessor

# Type definition for the configuration
ConfigDict = Dict[str, Union[str, List[str], Dict[str, List[str]]]]


class ProcessorIFD(BaseProcessor):
    def __init__(self) -> None:
        super().__init__()

        self.config: ConfigDict = {
            "target": "Fault Label",
            "numeric": [
                "Vibration (mm/s)",
                "Temperature (°C)",
                "Pressure (bar)",
                "RMS Vibration",
                "Mean Temp",
            ],
        }

    def process(self, input_files: List[Path], output_dir: Path) -> None:
        assert len(input_files) == 1, (
            "This processor is designed for a single file input."
        )

        input_file = input_files[0]

        df = pd.read_csv(input_file)
        X_train, X_test, y_train, y_test, ct = self.parse_and_split_tabular_data(
            df, self.config
        )

        # Save the processed data
        np.save(output_dir / "X_train.npy", X_train)
        np.save(output_dir / "X_test.npy", X_test)
        np.save(output_dir / "y_train.npy", y_train)
        np.save(output_dir / "y_test.npy", y_test)

        # Save the fitted ColumnTransformer for future use (e.g., inference)
        pd.to_pickle(ct, output_dir / "column_transformer.pkl")

    def parse_and_split_tabular_data(
        self,
        df: pd.DataFrame,
        config: ConfigDict,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_target: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
        """
        Splits the data safely before applying imputations and encodings to prevent data leakage.
        Returns: X_train, X_test, y_train, y_test, fitted_transformer
        """

        # 1. Identify Target
        target_col = config.get("target")
        if not target_col or target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")

        # 2. The Fixed Split (Before ANY math happens)
        stratify_param = df[target_col] if stratify_target else None

        df_train, df_test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=stratify_param
        )

        # Work on copies to prevent Pandas SettingWithCopy warnings
        df_train = df_train.copy()
        df_test = df_test.copy()

        y_train: np.ndarray = df_train[target_col].values
        y_test: np.ndarray = df_test[target_col].values

        # 3. Pre-process Datetime (Apply equally to both train and test)
        datetime_cols = config.get("datetime", [])
        extracted_time_features = []

        if isinstance(datetime_cols, list) and datetime_cols:
            for col in datetime_cols:
                # We must apply this to both dataframes independently
                for dataset in [df_train, df_test]:
                    dataset[col] = pd.to_datetime(dataset[col])
                    dataset[f"{col}_hour"] = dataset[col].dt.hour
                    dataset[f"{col}_dayofweek"] = dataset[col].dt.dayofweek

                extracted_time_features.extend([f"{col}_hour", f"{col}_dayofweek"])

        # 4. Build the Transformer Pipelines
        transformers: List[Tuple[str, Any, List[str]]] = []

        # Numeric
        numeric_cols = config.get("numeric", [])
        if isinstance(numeric_cols, list):
            all_numeric = numeric_cols + extracted_time_features
            if all_numeric:
                num_pipeline = Pipeline(
                    steps=[
                        ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
                        ("scaler", StandardScaler()),
                    ]
                )
                transformers.append(("num", num_pipeline, all_numeric))

        # Categorical
        categorical_cols = config.get("categorical", [])
        if isinstance(categorical_cols, list) and categorical_cols:
            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="constant", fill_value="missing"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
            transformers.append(("cat", cat_pipeline, categorical_cols))

        # Binary
        binary_cols = config.get("binary", [])
        if isinstance(binary_cols, list) and binary_cols:
            bin_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder()),
                ]
            )
            transformers.append(("bin", bin_pipeline, binary_cols))

        # Ordinal
        ordinal_config = config.get("ordinal", {})
        if isinstance(ordinal_config, dict) and ordinal_config:
            for col, mapping in ordinal_config.items():
                ord_pipeline = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(categories=[mapping])),
                    ]
                )
                transformers.append((f"ord_{col}", ord_pipeline, [col]))

        # 5. Initialize ColumnTransformer
        ct = ColumnTransformer(transformers=transformers, remainder="drop")

        # 6. Fit & Transform Train, ONLY Transform Test
        X_train: np.ndarray = np.array(ct.fit_transform(df_train))
        X_test: np.ndarray = np.array(ct.transform(df_test))

        return X_train, X_test, y_train, y_test, ct
