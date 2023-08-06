# Flight Price Prediction ✈️📈

## Introduction

This project aims to predict flight ticket prices using machine learning. The dataset contains flight ticket prices from March to June 2019, with 10,683 records in the training set and 2,671 records in the test set. By analyzing features like airline, source, destination, flight duration, and ticket prices, we develop an accurate prediction model. The results can provide insights for travelers and airlines to make informed decisions.

## Dependencies

To run this project, you'll need the following libraries:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
```

## EDA Findings

- Most preferred airline: "Jet Airways," followed by "Indigo."
- Most expensive tickets: "Jet Airways Business."
- Prices tend to be higher on weekends.
- "Total_Stops" and "flight duration" have high correlation with ticket prices.

## Model Building and Selection

Three models used for prediction: K Nearest Neighbors Regressor, Decision Tree Regressor, and Random Forest Regressor. 
Random Forest Regressor performs best, with 82% accuracy before hyperparameter tuning.

## Hyperparameter Tuning

RandomizedSearchCV applied to Random Forest Regressor, improving accuracy to 82%. 
The tuned model provides reliable forecasts for flight ticket fares.

## Connect with Me on LinkedIn 🤝

Feel free to connect with me on LinkedIn to discuss data science and machine learning.

## Feel Free to Use and Fork this Repo 🚀

This project is open-source. You are welcome to use, fork, and contribute to enhance 
its capabilities and benefit the aviation industry.

