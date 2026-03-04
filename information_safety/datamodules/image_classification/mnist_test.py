from information_safety.conftest import setup_with_overrides
from information_safety.datamodules.image_classification.image_classification_test import (
    ImageClassificationDataModuleTests,
)
from information_safety.datamodules.image_classification.mnist import MNISTDataModule


@setup_with_overrides("datamodule=mnist")
class TestMNISTDataModule(ImageClassificationDataModuleTests[MNISTDataModule]): ...
