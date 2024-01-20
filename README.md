# MobileSam Segmentation Model Service

A FastAPI service developed to deploy the MobileSam segmentation model. The service is containerized with Docker, to ensure easy setup and consistency across different environments. It can handle incoming requests asynchronously which can provide better performance and scalability. 

## Description

- **FastAPI** service defined in `app.py` exposes the model as a RESTful API. 
- **The POST endpoint `/segment-image`**: listens to incoming POST requests. When an image file is received, the service:
	- Validates the uploaded file to ensure that it is an image.
	- Reads the file's contents and converts the into  byte stream.
	- Opens the byte stream as an image object and coverts it into RGB format.
- **Model invocation**: The RGB image object is passed to the `segment_everything` function. This function utilizes the MobileSam model to process the image, performing segmentation, and returns the segmented image.
- **Response Formation**: 
	- The segmented image is converted into a byte array in PNG format.
	- A `Response` object is created with this byte array as it's content, and the media type is set to `image\png`.
	- This response, containing the segmented image in binary format, is sent back to the client.

## Setting Up and Running the Application

### Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed Docker on your machine. For installation instructions, see [Docker documentation](https://docs.docker.com/get-docker/)
- Add your user to the Docker group:

    ```bash
    sudo usermod -aG docker $USER
    ```

### Running the application

1. **Build your docker image**

    ```bash
    docker build -t my-fastapi-app .
    ```

2. **Run the image as a container:**

    ```bash
    docker run -d --name my-fastapi-container -p 8000:8000 my-fastapi-app
    ```

3. **Access the application**

    You can access the application at http://127.0.0.1:8000/docs#/ or http://localhost:8000/docs#/

## Testing

1. Automated testing

To run the automated tests for the API, navigate to the project root and execute:

```bash
    pytest
    ```


The test scripts are found in the directory `test`. The images used for testing are the directory `test/test_images`.

2. **Manual Testing with Postman or cURL**

- Using cURL:

    ```bash
    curl -X 'POST' 'http://127.0.0.1:8000/segment-image' -H 'accept: application/json' -H 'Content-Type: multipart/form-data' -F 'file=@tests/test_images/valid_image.jpg;type=image/jpeg' --output segmented_image_cURL.png
    ```	
- Using Postman
	- Open Postman and create a new request
	- Choose POST as the request method
	- Set the URL to `http://localhost:8000/segment-image`
	- Under the Headers tab, set Content-Type to multipart/form-data.
	- Under the Body tab, 
		- select the form-data option.
		- In the key field, enter `file`
		- Set the type to File from the dropdown on the right.
		- Use the Choose Files button to select an image file from your system.
	- Send the Request
	- Review the Response 

## Footnotes

- Fast API service, Docker setup and testing scripts built by [Davis Joseph](https://github.com/davisjoseph6)
