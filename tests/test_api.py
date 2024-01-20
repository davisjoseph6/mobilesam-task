import requests

API_URL = "http://localhost:8000/segment-image"

def test_segment_image_status_code():
    """Test that the API returns a 200 status code for a valid image file."""
    with open('tests/test_images/valid_image.jpg', 'rb') as file:
        response = requests.post(API_URL, files={'file': file})
        assert response.status_code == 200

def test_segment_image_response_format():
    """Test that the API returns data in the expected format."""
    with open('tests/test_images/valid_image.jpg', 'rb') as file:
        response = requests.post(API_URL, files={'file': file})
        # Replace the below line with the actual keys or data format you expect in response
        assert 'segmented_image' in response.json()

def test_segment_image_with_invalid_file():
    """Test the API with an invalid file to check error handling."""
    with open('tests/test_images/invalid_file.txt', 'rb') as file:
        response = requests.post(API_URL, files={'file': file})
        # Adjust the status code and message based on your API's error handling
        assert response.status_code == 400
        assert response.json()['detail'] == 'Invalid file format.'

