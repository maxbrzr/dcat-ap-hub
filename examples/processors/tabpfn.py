from typing import Any

from numpy import ndarray

from dcat_ap_hub.core.interfaces import SKLearnModel
from dcat_ap_hub.internals.utils import from_import_or_install


class TabPFNModel(SKLearnModel):
    def __init__(self) -> None:
        super().__init__()

        TabPFNRegressor = from_import_or_install("tabpfn", "tabpfn", "TabPFNRegressor")
        ModelVersion = from_import_or_install(
            "tabpfn", "tabpfn.constants", "ModelVersion"
        )
        self.clf = TabPFNRegressor.create_default_for_version(ModelVersion.V2)

    def fit(self, X_train: ndarray, y_train: ndarray) -> None:
        return self.clf.fit(X_train, y_train)

    def predict(self, X_test: ndarray) -> Any:
        return self.clf.predict(X_test)
