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

## Model Training

- After finding from EDA, few best ML model to model this dataset is Prophet, LSTM and GRU. However, Prophet might not be best tool for this exercise as it come with a set of default parameters and heuristics that often work well out-of-the-box.
- LSTM, GRU and Prophet was trained. 
- LSTM was able to catch the weekly pattern but failed to capture the yearly pattern (always increasing), however GRU was able to catch both yearly and weekly pattern but after around July 2022 it repeats the same patttern. So, GRU was choosen.
- Although, Prophet might not be helpful to access the ML skill it was also trainned and saved because with data of 2021 it was able to model the yearly data of 2022. However, it wasn't able to model the weekly pattern.
- Using the output of both GRU and prophet, would be helpful to predict the receipt count of 2022. However, I believe that if we had data set of one more year only GRU (or similar) model would to enough for the prediction.

## Project Organization

## Conclusion
