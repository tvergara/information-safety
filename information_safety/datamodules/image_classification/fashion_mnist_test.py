from information_safety.conftest import setup_with_overrides
from information_safety.datamodules.image_classification.fashion_mnist import FashionMNISTDataModule
from information_safety.datamodules.image_classification.image_classification_test import (
    ImageClassificationDataModuleTests,
)


@setup_with_overrides("datamodule=fashion_mnist")
class TestFashionMNISTDataModule(ImageClassificationDataModuleTests[FashionMNISTDataModule]): ...
