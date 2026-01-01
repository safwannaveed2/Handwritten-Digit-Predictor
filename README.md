MNIST Handwritten Digit Predictor

A simple and interactive web application to predict handwritten digits using a pre-trained neural network. Built with Streamlit and TensorFlow/Keras, the app allows users to upload an image of a digit and get an instant prediction.

🔹 Features

Upload a handwritten digit image (PNG, JPG, JPEG).

Pre-trained neural network predicts the digit instantly.

Lightweight and easy-to-use interface.

No training or configuration needed by the user.

🛠️ Technologies Used

Python 3.10+

Streamlit – For web interface

TensorFlow / Keras – Neural network model

NumPy – Data processing

Pillow – Image processing

🚀 How to Run

Clone the repository:

git clone https://github.com/yourusername/mnist-digit-predictor.git
cd mnist-digit-predictor


Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app.py


Upload a handwritten digit image and see the predicted result instantly.


🧠 Model Architecture

Input Layer: 784 neurons (28x28 flattened image)

Hidden Layer 1: 512 neurons, ReLU activation

Hidden Layer 2: 256 neurons, ReLU activation

Output Layer: 10 neurons, Softmax activation

📂 Project Structure
mnist-digit-predictor/
│
├─ app.py               # Streamlit application
├─ requirements.txt     # Project dependencies
├─ README.md            # Project documentation
