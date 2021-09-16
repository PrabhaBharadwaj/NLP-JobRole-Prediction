# NLP-JobRole-Prediction

# JOB ROLE PREDICTION
- The core of HR products often involve APIs around machine learning algorithms and NLP.
- In real world data especially in human resources, they come in different schemas and it is important to standardize them into one.
- For this problem, we will be looking into classifying any job titles into job functions. Initially, we tried to train a model that classify job titles into all job functions. However, we found out that the information technology job functions are too general and it is important for us to break them further down into various subclasses.
- By breaking them down, we are then able to match candidates to jobs to a higher accuracy.
- This problem will test your ability to build a basic NLP model based on a given dataset.
- Your end task will be to: - develop a model to predict one of the 16 classes (see Variables Schema); - provide justification for model evaluation and report your results; and
deploy your model in the form of an API endpoint (any API framework will do, but FastAPI is preferred).

- You will be assessed on:

    - your ability to build a model pipeline and deploy it as an API (50%);
    - model accuracy (10%); and
    - writing clean, readable code (40%)
 
 
#### Column Name - Description
**id**
  - A unique identifier for every job title. This is purely for our reference. Do not use it at all.
  
**Job Title**
  - Job Title scraped from the job description. Do take note that this data is unclean and may consist of unnecessary field. In various job titles, you will be able to see the duration as well. 
  - This column is your X label.
  
**Type**
  - Job Function of a job. This column is your Y label. It consists of the following 16 classes. 
  - Non - IT, Backend Engineer, Project Management, Product Management, Customer Support,Design, Data Science, Full Stack Engineer, Technical Support, Front End Engineer, Data  Analyst, Mobile Application Developer, Database Administration, Cloud architect, Information Security,
  
#### Results
Submit an updated test file with a new third column called "Type". The type will contain one of the 16 classes


##### Note: Here "text_classification.joblib.BZ2" is huge file so ITS NOT uploaded in github, Due to this missing joblib file, couldnot deploy in Heroku.
But can run this below file in local to see the API

uvicorn app:app --reload

Path: 
http://127.0.0.1:8000/docs

![image](https://user-images.githubusercontent.com/66779952/133581801-d2a0319b-dedd-4516-81c9-eb103d5fa93d.png)



![image](https://user-images.githubusercontent.com/66779952/133581891-32f98e04-4f21-4f9a-9779-015247c0cd18.png)




-------------------------------------------------------------------------


# STEPS:

### 1.Created 2 Jupiter notebook :

In 1st one I have done all prepossessing and trained with different classification model
In 2nd notebook, applied same prepossessing and created pipeline and applied only best model which I had decided in 1st notebook. Created joblib file

#### a)1_JobTitle_Classification_Task2_Model:

1.Here we applied different data prepossessing 
        Data Analysis(EDA)/ Data Cleaning /Feature Engineering 
        Tokenization
        Lower case convertion
        Digits Removal
        Unicodedata removal
        Lemmatization
        Stop word removal
        Single character word removal
        Rare word removal etc
        Label encoding to target field
			
        Converted Text to Numerical field
  				  - BOW, TFIDF etc

##### 2.Model Building: Trained with different model  

        SGDClassifier
        LogisticRegression
        LinearSVC
        RandomForestClassifier
        GaussianNBS

##### 3.Model Evaluation
This is Classification problem so used different Classification related evaluation matrix
        Accuracy
        Miss_class_rate
        Precision
        Recall
        f1

##### 4.Good accuracy model is used as the final model

##### 5.Apply same model to Test data and Create Final Submission file. Created X0PA_DS_TEST_RESULT.csv” file which having predicted type for test data


#### 2.2_JobTitle_Classification_Task2_Joblib:

##### 1.Here we applied different data prepossessing 
          Data Analysis(EDA)/ Data Cleaning /Feature Engineering 
          Tokenization
          Lower case convertion
          Digits Removal
          Unicodedata removal
          Lemmatization
          Stop word removal
          Single character word removal
          Rare word removal etc

          Converted Text to Numerical field
                      - TFIDF etc

##### 2.Model Building: Trained with best model  

    RandomForestClassifier

##### 3.Model Evaluation
This is Classification problem so used different Classification related evaluation matrix
        Accuracy
        Miss_class_rate
        Precision
        Recall
        f1

##### 4.Created text_classification.joblib file



#### 3.FastAPI Build:

      Created app.py file and imported all necessary files
      Created supporting document like 
      Procfile (To deploy in Heroku)
      Requirements (To install all necessary files)

      Created LOCAL environment and activated this

            conda create -n role_prediction python=3.6

            activate role_prediction 


      Ran requirement.txt in local to install all library
          pip install -r requirements.txt

      Ran app.py in local to test the code.

        uvicorn app:app --reload

      Copied below link to test app and imputed the string
      http://127.0.0.1:8000/docs

#### 4.Check in all code in Github:

https://github.com/PrabhaBharadwaj/NLP-JobRole-Prediction

#### 5.Deployed code via Heroku :

Here "text_classification.joblib.BZ2" is huge file so ITS NOT uploaded in github, Due to this missing joblib file, couldnot deploy in Heroku.
But can run this below file in local to see the API
