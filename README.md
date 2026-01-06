

# Telecom Customer Churn & LTV Prediction Dashboard


A machine learning-powered web app built with Streamlit to predict customer churn, lifetime value (LTV), and behavioral segmentation for a telecom company.

## 🚀 Features

* Churn Prediction: Uses a classification model to predict the probability (%) that a customer will churn.
* Lifetime Value (LTV) Prediction: Uses a regression model to estimate the total revenue a customer will generate.
* Customer Segmentation: Uses a clustering model to assign customers to a predefined persona (e.g., "Loyal Champion", "High-Value, At-Risk").
* Interactive Interface: An easy-to-use form to input customer details and get real-time predictions.

## 🛠️ How to Run

Follow these steps to get the app running locally.
````
**1. Clone the Repository**
```bash
git clone [https://github.com/MrToshith/telecom-customer-churn-ltv-prediction.git](https://github.com/MrToshith/telecom-customer-churn-ltv-prediction.git)
cd telecom-customer-churn-ltv-prediction

````
**2. Create a Virtual Environment (Recommended)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies**
Make sure you have your `requirements.txt` file in the directory.

```bash
pip install -r requirements.txt
```

**4. Run the Streamlit App**

```bash
streamlit run app.py
```

The app will open automatically in your browser.

## 📦 Project Structure

To run this project, you must have the following files in your root directory:

  * `app.py`: The main Streamlit application script.
  * `requirements.txt`: A list of all required Python libraries.
  * `customer_data_with_personas.csv`: The dataset used for model training (and needed here for feature reference).
  * `classification_model.joblib`: The pre-trained churn prediction model.
  * `regression_model.joblib`: The pre-trained LTV prediction model.
  * `cluster_model.joblib`: The pre-trained segmentation model.
  * `classification_scaler.joblib`: The scaler for the classification model.
  * `regression_scaler.joblib`: The scaler for the regression model.
  * `cluster_scaler.joblib`: The scaler for the clustering model.

## Libraries Used

This project relies on the following main libraries:

  * [Streamlit](https://streamlit.io/)
  * [Pandas](https://pandas.pydata.org/)
  * [Numpy](https://numpy.org/)
  * [Scikit-learn](https://scikit-learn.org/) (for `joblib` and model execution)

Link : https://telecom-customer-churn-ltv-prediction.streamlit.app/

<!-- end list -->

```
```
