# test_segmentation_api.py or conftest.py
import pytest
import requests

@pytest.fixture
def url():
    return "http://localhost:8000/segment-image"

@pytest.fixture
def image_path():
    return 'tests/test_images/valid_image.jpg'  

def test_segment_image_api(url, image_path):
    with open(image_path, 'rb') as file:
        files = {'file': (file.name, file, 'image/jpeg')}
        response = requests.post(url, files=files)
        assert response.status_code == 200

