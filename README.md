# Cloud-Based Satellite Image Classifier

This project is an end-to-end cloud-based pipeline for classifying satellite imagery. It uses a Convolutional Neural Network (CNN) model built with PyTorch, served by a Flask web application, and deployed on an AWS EC2 instance.

![Screenshot of the Application](./screenshot.png)
*(To add a screenshot: take a picture of your running application, name it `screenshot.png`, and upload it to your repository. This line will then display it.)*

---

## Features

- **AI-Powered Classification:** Utilizes a PyTorch CNN model to classify satellite images into four categories: `Cloudy`, `Desert`, `Green Area`, and `Water`.
- **Web Interface:** A clean and responsive user interface built with Flask and HTML/CSS allows users to easily upload an image for analysis.
- **Cloud Deployment:** Fully deployed on an AWS EC2 instance, making the service accessible from anywhere.
- **Scalable Backend:** Uses Gunicorn as a production-ready WSGI server to handle concurrent requests efficiently.

---

## Tech Stack

- **Backend:** Python, Flask
- **Machine Learning:** PyTorch, Pillow, Torchvision
- **Deployment:** AWS EC2, Gunicorn
- **Frontend:** HTML5, CSS3, JavaScript

---

## Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/satellite-image-classifier.git
    cd satellite-image-classifier
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file for this step to work. See the next section.)*

4.  **Run the Flask application:**
    ```bash
    python app.py
    ```
    The application will be available at `http://127.0.0.1:5000`.

---

## How to Use

1.  Navigate to the web interface.
2.  Click the upload box to select a satellite image file (`.jpg`, `.png`, etc.).
3.  The application will process the image and display the predicted class along with a confidence score breakdown.
