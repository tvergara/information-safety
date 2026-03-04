from information_safety.conftest import setup_with_overrides
from information_safety.datamodules.image_classification.cifar10 import CIFAR10DataModule
from information_safety.datamodules.image_classification.image_classification_test import (
    ImageClassificationDataModuleTests,
)


@setup_with_overrides("datamodule=cifar10")
class TestCIFAR10DataModule(ImageClassificationDataModuleTests[CIFAR10DataModule]): ...
