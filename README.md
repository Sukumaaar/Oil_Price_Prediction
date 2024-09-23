# Project Overview
This thesis project focuses on predicting oil price volatility using advanced deep learning models such as Long Short-Term Memory (LSTM), Gated Recurrent Units (GRU), and CNN-LSTM. The research explores how these models perform in comparison to traditional forecasting methods, leveraging a combination of macroeconomic data, geopolitical risk indices, and oil-specific data to enhance the predictive accuracy of oil prices.

##Objectives
To predict oil price volatility using deep learning models that can capture temporal dependencies in the data.
To evaluate the performance of LSTM, GRU, and CNN-LSTM models in terms of prediction accuracy, comparing them to traditional time-series forecasting models.
To analyze the influence of macroeconomic factors and geopolitical risks on oil price movements and incorporate them into the prediction models.
Dataset
The dataset used in this project contains three primary types of data:

Macroeconomic Indicators: GDP, inflation, unemployment rates, and interest rates from major oil-producing and consuming countries.
Geopolitical Risk Indices: Geopolitical Risk Index (GPR), Geopolitical Risk Threat (GPRT), and Geopolitical Risk Action (GPRA) to account for the impact of geopolitical events.
Oil-Specific Data: Daily West Texas Intermediate (WTI) oil prices.
The data spans multiple years, providing a rich set of historical data for training the models. It was collected from sources like the World Bank, geopolitical risk indices, and oil price datasets.

Data Preprocessing
Data preprocessing steps included:

Handling missing values using interpolation.
Normalization to scale the features.
Lagging variables to capture historical dependencies.
Outlier detection and removal for cleaner data.
Models Used
Three deep learning models were implemented:

LSTM (Long Short-Term Memory): Effective for capturing long-term dependencies in sequential data.
GRU (Gated Recurrent Units): A simplified version of LSTM that is computationally less intensive.
CNN-LSTM: Combines Convolutional Neural Networks (CNN) and LSTM to capture both spatial and temporal patterns in the data.
Model Training and Evaluation
The dataset was split into training (80%) and testing (20%) sets.
The models were evaluated based on metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²).
Results
The evaluation metrics showed that deep learning models, particularly the CNN-LSTM, outperformed traditional models in predicting oil price volatility. However, simpler models like GRU also performed reasonably well, offering a good trade-off between accuracy and computational efficiency.

Tools and Libraries
The project was developed using the following tools and libraries:

Python: The primary programming language.
TensorFlow and Keras: For implementing deep learning models.
Pandas and NumPy: For data manipulation and analysis.
Matplotlib and Seaborn: For data visualization.
Scikit-learn: For traditional machine learning models and evaluation metrics.
Conclusion
This project demonstrates that deep learning models can significantly improve the accuracy of oil price volatility predictions. CNN-LSTM, with its ability to handle complex temporal and spatial patterns, emerged as the best performer in this study. The inclusion of macroeconomic and geopolitical risk factors further enhanced the models' ability to predict price movements.

Future Work
Future improvements could involve:

Incorporating real-time data streams for up-to-date predictions.
Experimenting with other hybrid models or ensemble techniques.
Expanding the dataset with more external factors such as environmental policies and global trade dynamics.
