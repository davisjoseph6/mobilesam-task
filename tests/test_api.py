import requests

API_URL = "http://localhost:8000/segment-image"

def test_segment_image_status_code():
    """Test that the API returns a 200 status code for a valid image file."""
    with open('tests/test_images/valid_image.jpg', 'rb') as file:
        files = {'file': ('valid_image.jpg', file, 'image/jpeg')}
        response = requests.post(API_URL, files=files)
        assert response.status_code == 200

def test_segment_image_response_format():
    """Test that the API returns data in the expected format (an image)."""
    with open('tests/test_images/valid_image.jpg', 'rb') as file:
        files = {'file': ('valid_image.jpg', file, 'image/jpeg')}
        response = requests.post(API_URL, files=files)
        assert response.status_code == 200
        assert response.headers['Content-Type'] == 'image/png'  

def test_segment_image_with_invalid_file():
    """Test the API with an invalid file to check error handling."""
    with open('tests/test_images/invalid_file.txt', 'rb') as file:
        files = {'file': ('invalid_file.txt', file, 'text/plain')}
        response = requests.post(API_URL, files=files)
        assert response.status_code == 400
        assert response.json()['detail'] == 'File is not an image or content type is missing.'  

