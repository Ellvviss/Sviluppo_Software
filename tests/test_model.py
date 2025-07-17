import os, shutil
import pytest
from src.ai import eda, train_save, evaluate, download_data

RESULTS : str = "./results"
MODELPATH_SAVE : str = "./results/dogs_vs_cats_model.pth"

@pytest.fixture(autouse=True)
def setup_and_cleanup():
    download_data()
    yield
    shutil.rmtree("./results/", ignore_errors=True)
    shutil.rmtree("./data/", ignore_errors=True)
    shutil.rmtree("./persistent_data/", ignore_errors=True)

def test_eda():
    eda()
    assert os.path.exists(os.path.join(RESULTS, "class_distribution.png")), "Class distribution plot missing"
    assert os.path.exists(os.path.join(RESULTS, "samples_images.png")), "Sample images plot missing"

def test_tain_save():
    train_save(model_path=RESULTS, epochs=1)
    assert os.path.exists(MODELPATH_SAVE), "Model file not found after training."
    assert os.path.getsize(MODELPATH_SAVE) > 0, "Model file is empty."

def test_evaluate():
    train_save(model_path=RESULTS, epochs=1)
    evaluate()
    assert os.path.exists(os.path.join(RESULTS, "confusion_matrix.png")), "Confusion matrix plot missing after evaluation."