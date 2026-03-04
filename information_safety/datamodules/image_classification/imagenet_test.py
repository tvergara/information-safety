import pytest

from information_safety.conftest import setup_with_overrides
from information_safety.datamodules.image_classification.image_classification_test import (
    ImageClassificationDataModuleTests,
)
from information_safety.datamodules.image_classification.imagenet import ImageNetDataModule
from information_safety.utils.testutils import needs_network_dataset_dir


@pytest.mark.slow
@needs_network_dataset_dir("imagenet")
@setup_with_overrides("datamodule=imagenet")
class TestImageNetDataModule(ImageClassificationDataModuleTests[ImageNetDataModule]): ...
