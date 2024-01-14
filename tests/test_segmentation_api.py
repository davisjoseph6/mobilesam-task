import requests

def test_segment_image_api(url, image_path):
    files = {'file': open(image_path, 'rib')}
    response = requests.post(url, files=files)

    if response.status_code == 200:
        print("API call successful. Saving the segmented image.")
        with open('segmented_image.png', 'wb') as f:
            f.write(response.content)
        print("Segmented image saved as segmented_image.png")
    else:
        print("Error in API call:", response.status_code, response.txt)

if __name__ == "__main__":
    api_url = 'http://127.0.0.1:8000/segment-image' # Update with the correct URL if needed
    image_path = 'path_to_test_image.jpg' # Update with the path to your test image
    test_segment_image_api(api_url, image_path)
