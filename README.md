# Drive Guardian 🚗: Automatic Breaking Assistant
![image](images/main.png)


## ⭐⭐ Project Showcase at Hackaccino 2023, Bennett University


Developed by [Dhruv Kumar](https://github.com/DhruvK278) and [Rohan Pandey](https://github.com/rohan-pandeyy)

### Video Demo




## Table of Contents
- [Background](#background)
- [Features](#features)
- [ML Algorithm](#ml-algorithm)
- [Instructions to Run](#instructions-to-run)
- [Requirements](#requirements)
- [Description of Project](#description-of-project)
- [Contributions and Acknowledgments](#contributions-and-acknowledgments)

## Background
With the increasing number of road accidents, Drive Guardian aims to enhance driver safety by providing a tool that automatically applies brakes when necessary. Our application leverages advanced machine learning algorithms, such as YOLO and lane detection, to ensure precise brake timing, reducing the risk of accidents.
 
## Description of Project
Our project aims to enhance the safety of drivers and passengers alike by leveraging live feeds from dashcam videos and utilizing advanced algorithms for real-time analysis of applying brakes.

## Features
- Real-time Analysis: Our system provides sophisticated real-time analysis of road conditions, continuously monitoring for potential hazards.
- Object Detection and Prediction: Utilizing advanced algorithms, we detect objects and predict potential hazards, empowering drivers with proactive insights.
- Enhanced Video Clarity: Through seamless integration of upscaling techniques, we enhance the clarity and accuracy of the video feed for precise analysis.

  
<img width="968" src="images/howto.png">

## ML Algorithm
<img width="969" src="images/ML.png">
<img width="969" src="images/img4.jpeg">

### Input
User-provided dashcam video or stream with their device.

### Data Preprocessing


### Sentiment Analysis Techniques
- Utilized advanced NLP techniques, including sentiment lexicons and machine learning models.
- Developed an ML model to classify textual data into positive, negative, or neutral sentiment categories.
- Leveraged NLP models from `Hugging Face transformers` to achieve exceptional accuracy in sentiment classification.
- Utilized `VADER` (NLP-based model) for bulk data analysis.
- Trained a `naive Bayes classifier` specifically for analyzing reviews.

### Dataset
- Size ranges from 40,000 to 50,000 samples.

### Output
Sentiment scores and visualizations depicting sentiment trends.


## Instructions to Run
To run this project locally, follow these steps:

- Install Requirements
```pip install -r requirements.txt```

- Run the application
```streamlit run Home.py```

## Requirements
```
pandas
numpy
nltk
scikit-learn
streamlit
matplotlib
plotly==5.17.0
pygwalker==0.4.8
scipy==1.13.0
st_annotated_text==4.0.1
stqdm==0.0.5
streamlit==1.26.0
streamlit_extras==0.4.2
transformers==4.33.2
wordcloud==1.9.2
torch==2.2.2
```

## Contributions and Acknowledgments
This project is open for contributions, and we welcome any feedback or suggestions for improvement. If you find this project useful, feel free to use it for your needs. When attributing this project, please mention:
```
InSocial by Samaksh Tyagi & Sukant Aryan
Repository: https://github.com/samakshty/insocial
```