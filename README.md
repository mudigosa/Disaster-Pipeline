# Disaster-Pipeline
Disaster Response Pipeline Project 

1. Installation  
I use python 3.5 to create this project and the main libraries I used are:     
    Python 3.5+ (I used Python 3.7)
    Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
    Natural Language Process Libraries: NLTK
    SQLlite Database Libraqries: SQLalchemy
    Web App and Data Visualization: Flask, Plotly

Please check detailed version information in requirement.txt.
2. Project Motivation

I create a machine learning pipeline to categorize these events so that I can send the messages to an appropriate disaster relief agency. I split the data into a training set and a test set. Then,I create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model. Finally, I export the model to a pickle file.

 The web app also displays visualizations of the data as follows: disaster graph1
3. File Descriptions

    \
        README.md
        ETL disaster Pipeline Preparation.ipynb
        ML disaster Pipeline Preparation.ipynb
    \data
        disaster_categories.csv
        disaster_messages.csv
    \models
        classifier.pkl : It is too big(about 2GB size) to be included in the github. To run ML pipeline that trains classifier and saves the trained model to MachineLearningPipeline.py

4.Instructions:

 1. Run the following commands in the project's root directory to set up your database and model.

     - To run ETL pipeline that cleans data and stores in database
         `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterMessages.db`
     - To run ML pipeline that trains classifier and saves
         `python models/MachineLearningPipeline.py data/DisasterResponse.db models/classifier.pkl`

 2. Run the following command in the app's directory to run your web app.
     `python run.py`

5. GitHub link:

    https://github.com/mudigosa/Disaster-Response-Pipelines

6. Licensing, Author, Acknowledgements

This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License. Please refer to Udacity Terms of Service for further information.
