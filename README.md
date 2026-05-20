## Bangladesh Air Quality Index (AQI) Predictor

An interactive Deep Learning web application built using an Artificial Neural Network (ANN) to predict the real-time Air Quality Index (AQI) across various regions in Bangladesh. 
The model analyzes key geographical coordinates and atmospheric pollutant concentrations to deliver highly accurate regression predictions, wrapped beautifully in a Streamlit user interface.

## Project Performance
* **R-squared ($R^2$) Score:** `0.9108` (Explains over 91% of variance in the data)
* **Mean Absolute Error (MAE):** `10.78` AQI Points
* **Custom Prediction Accuracy:** `70.37%` (Within strict error boundaries)

## Tech Stack used in this project
* **Deep Learning Framework:** TensorFlow / Keras
* **Data Processing:** Python, NumPy, Pandas, Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn
* **Web Interface:** Streamlit

## The Engineering Journey (Fine-Tuning Process)
1. **Data Preprocessing & Cleaning:** Handled over 1 Million rows of data, removed missing values, and isolated geographical and chemical features (`pm10`, `pm2_5`, `carbon_monoxide`, `nitrogen_dioxide`, `sulphur_dioxide`, `ozone`).
2. **Feature Engineering:** Identified highly skewed features using histograms and applied a `log1p` mathematical transformation to normalize distributions. Scaled features using a `StandardScaler`.
3. **Architecture Optimization:** Corrected an initial regression-to-classification mismatch by refactoring the final layer to a single linear continuous node (`Dense(1)`), updating the optimization goal to Mean Squared Error (`mse`).
4. **Hyperparameter Tuning:** Fine-tuned learning rates, structural neuron depth ($256 \to 128 \to 64$), and integrated `Dropout` regularization layers along with `EarlyStopping` to achieve textbook-perfect validation learning curves without overfitting.

## Hey wanna run this locally?
1. Clone this repository:
```bash
git clone [https://github.com/raocious/AQI.git](https://github.com/raocious/AQI.git)
