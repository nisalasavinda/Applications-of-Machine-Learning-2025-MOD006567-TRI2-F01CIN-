❤️ Cardiovascular Disease Prediction System

This project is an AI-powered application designed to predict cardiovascular disease (CVD) risk using clinical data and ECG signals. It integrates machine learning, deep learning, and AI-generated medical reporting to assist in early diagnosis and decision-making.



🚀 Features

* 📊 Clinical data-based heart disease prediction
* 📈 ECG signal/image-based analysis using deep learning
* 🤖 AI-generated medical report using Google Gemini
* 🌐 Interactive web interface built with Streamlit
* 📉 Data visualization and risk interpretation



⚠️ API Key Notice

The original Google Gemini API key used during development has been **removed for security reasons**.

To enable AI report generation, add your own API key in `app.py`:

python
API_KEY = "your_api_key_here"




📦 Required Python Libraries

The project depends on the following libraries:

* streamlit
* numpy
* pandas
* scikit-learn
* matplotlib
* seaborn
* tensorflow
* keras
* opencv-python
* pillow
* google-generativeai
* joblib



🔧 Installation


Install dependencies:

bash
pip install -r requirements.txt


Run the application:


streamlit run app.py



🧠 Technologies Used

* Machine Learning (Scikit-learn)
* Deep Learning (TensorFlow, Keras)
* Computer Vision (OpenCV)
* AI Report Generation (Google Gemini)
* Web Framework (Streamlit)



📌 Notes

* Ensure Python 3.9+ is installed
* Add your own Gemini API key before running
* Models should be pre-trained and saved before deployment



📬 Future Improvements

* Real-time ECG signal integration
* Explainable AI (XAI) for better interpretability
* Deployment on cloud platforms
* Integration with wearable health devices

