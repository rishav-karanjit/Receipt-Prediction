# Receipt-Prediction

## To run

> Please let me know if you are unable to run the code.
 
#### 2 Ways to run:

1. Using Docker:
- Pull docker image
```docker pull rishavkaranjit/dockerhub:receipt_prediction_rishav```
- Run docker image
```docker run -p 5000:5000 rishavkaranjit/dockerhub:receipt_prediction_rishav```
- Visit the url shown on the console
2. Cloning this repository:
  - Clone this repository
  - Install requirements:
```pip install -r requirements.txt```
  - Run main.py
    ```python main.py```
  - ```Note: This codes were compiled and executed on windows machine with python 3.9```
     
## Exploratory Data Analysis (EDA)

- Before getting started the ML model, the data analysis was performed on the data.
- Summary of EDA findings:
  - The dataset is complete without any missing values.
  - A clear trend and seasonality are present in the data, with potential weekly and year wide patterns.
  - The receipt count seems to be in an increasing in the year wide pattern.
  - Weekly patterns suggest higher receipt counts at the beginning of the week, decreasing towards the weekend.
  - Autocorrelation analysis indicates potential weekly seasonality and influence from recent days.
- Reference:
  - [Exploratory Data Analysis (EDA) directory](https://github.com/rishav-karanjit/Receipt-Prediction/tree/main/EAD)

## Model Training

- After finding from EDA, few best ML model to model this dataset is Prophet, LSTM and GRU. However, Prophet might not be best tool for this exercise as it come with a set of default parameters and heuristics that often work well out-of-the-box.
- LSTM, GRU and Prophet was trained. 
- LSTM was able to catch the weekly pattern but failed to capture the yearly pattern (always increasing), however GRU was able to catch both yearly and weekly pattern but after around July 2022 it repeats the same patttern. So, GRU was choosen.
- Although, Prophet might not be helpful to access the ML skill it was also trainned and saved because with data of 2021 it was able to model the yearly data of 2022. However, it wasn't able to model the weekly pattern.
- Using the output of both GRU and prophet, would be helpful to predict the receipt count of 2022. However, I believe that if we had data set of one more year only GRU (or similar) model would to enough for the prediction.
- LSTM model training is also added just for the record in this repo. LSTM model training code is not available on docker.
- The conventional method selecting model would be using performance metrics (train, validation and test set). However, this was not done because of very less dataset available. The data was not divided into test set thinking that the models will not have enought data of train if it was separated. So, the pattern the model predicts for 2022 was seen and the best model which was able to capture 2021 pattern and predict 2022 was selected.
- Reference:
  - [Model training directory](https://github.com/rishav-karanjit/Receipt-Prediction/tree/main/Model%20Training%20Code)
  - GRU Training Screenshot
    - ![GRU Training Screenshot](https://github.com/rishav-karanjit/Receipt-Prediction/blob/main/Project%20Images/GRU%20training.png)
  - GRU Model Summary
    - ![GRU Training Summary](https://github.com/rishav-karanjit/Receipt-Prediction/blob/main/Project%20Images/GRU%20summary.png) 

## Self reflection and Conclusion

### Self Reflection

What went well?
- Data is the foundation of machine learning. So, EDA was performed prior to training of ML models.
- I trained 3 models and was able to select 2. Both of the model was able to model 2021 data and predict 2022 data.
- ML models were manually hypertuned with different parameters to select the best model.
- Every possible techniques/tools were used to select parameter. Some technique/tool were Scaler, EarlyStopping, Validation Split and Plotly (to plot).
- To maximize model training, conventional method of separating data into test set was not used because the models already lacked data to be trained with. The idea to not use test set came from my experience with Machine Learning.
- Comments were added wherever possible.
- These model were saved and a web application was created to visualize with plot on these models.
- The code of web application was organized into various files so that the next dev who works on this code will have easy onboarding to the codebase
- The documentation is written in a best possible way to help the next developer to onboard on the codebase.

What didn't go well with reasons
- Test set was not created. So, we don't have conventional evidence that the model is not overfitting. However, the plots of 2022 prediction clearly shows that the model was not fitting. The model was able to capture pattern from 2021 data and apply the same pattern to 2022.
- Unit testing was not done in the web application. This is because this project will not go to the production environment and spending extra dev time for unit testing was thought to be redundant. Another reason is that, in my past experience web application would be created to present the output to product owner and unit testing was not done there.
- Going through a code review process would be very helpful as I would get the second person's feedback which could improve my code.

### Conclusion 

The dataset didn't had any missing values which was great. However, if we had a dataset of one more year the model could train better. 
