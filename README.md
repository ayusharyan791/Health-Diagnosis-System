# Health Diagnosis using AI Chatbot

## Overview
This project implements a Health Diagnosis system using an AI chatbot. The chatbot assists users in determining potential health issues based on symptoms provided by the user. The system utilizes a Decision Tree classifier for prognosis prediction, and it provides information about the predicted condition, severity, precautions, and suggests consulting a doctor.

## Files and Directories

- *app.py*: Flask web application that integrates the chatbot functionality into a web interface.
- *index.html*: HTML file containing the user interface for the chatbot.
- *MasterData/*: Directory containing master data files for symptom descriptions, severity, and precautions.
- *Data/*: Directory containing training and testing datasets.
- *healthcare-chatbot/Data/Training.csv*: CSV file containing training data for the machine learning model.
- *healthcare-chatbot/Data/Testing.csv*: CSV file containing testing data for the machine learning model.
- *healthcare-chatbot/Data/doctors_dataset.csv*: CSV file containing information about doctors associated with specific diseases.

## Prerequisites
- Python 3.x
- Flask
- pandas
- scikit-learn
- numpy
- pyttsx3

Install the required packages using the following command:
```bash
pip install flask pandas scikit-learn numpy pyttsx3
