# Internship Take Home Assignment - Software Engineer

This assignment is designed to assess your software engineering skills in the context of integrating and deploying a machine learning model. Though the task involves a machine learning model, the primary focus is on developing and deploying the software application.

## How to Run the Code

1. **Create a Virtual Environment and Activate It**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. **Install the Requirements**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Code**

    ```bash
    python main.py
    ```

## Task Overview

**Title:** MobileSam Segmentation Model Service

**Expected Time to Complete:** 4 hours

**Objective:** Develop a FastAPI service to deploy the MobileSam segmentation model, containerize the service with Docker, and ensure efficient interaction with the model on the CPU.

**Background:**
MobileSam is a machine learning model specialized in image segmentation on CPUs. Your task is to create a microservice that allows users to interact with this model via an API. You should find the script `main.py` in this repository, which contains the MobileSam model and a function `segment_everything` that takes an image as input and returns the segmentation result. You can use this function to develop your service. Ignore the default parameters of the function for now.

## Task Description

- **Develop a Microservice:** Use a Python API framework (we suggest FastAPI) to expose the MobileSam segmentation model as a RESTful API.
  
- **Model Integration:** Incorporate the MobileSam segmentation model into your service. It should process image inputs and return segmentation results.
  
- **API Endpoints:** Create a POST endpoint `/segment-image` to accept an image file, process it through MobileSam, and return the segmentation result.
  
- **Documentation:** Provide clear instructions for setting up, running, and interacting with the service in a README.md file.

- **[Bonus]** Docker Familiarity: Containerize your FastAPI service using Docker.

## Submission

- Submit your code via a GitHub repository link.
- Include a README file with detailed setup and usage instructions.
- Provide any necessary scripts or files for testing the API.
