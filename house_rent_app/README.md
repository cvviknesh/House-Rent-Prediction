# 🏠 RentIQ — House Rent Prediction App

A production-grade Streamlit web application that predicts monthly house rent across 6 major Indian cities using a Random Forest Machine Learning model.

**Live demo:** Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) for free.

---

## Features

- Predicts monthly rent based on BHK, size, city, furnishing, floor, and tenant type
- Smart insight badges (above/below city average, metro premium, etc.)
- City average comparison table
- Clean, professional UI with custom styling
- Trained on 4,746 real Indian property listings (Kaggle dataset)

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest Regressor |
| R² Score | ~0.84 |
| Features | BHK, Size, Floor, Area Type, City, Furnishing Status, Tenant Preferred, Bathroom |
| Training split | 85% train / 15% test |
| Cross-validation | 5-fold |

## Project Structure

```
house_rent_app/
├── app.py              # Main Streamlit application
├── model_trainer.py    # ML model training and saving
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

The model trains automatically on first run and is cached for future use.

## How to Deploy on Streamlit Cloud (Free)

1. Push this folder to your GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select `app.py` as the main file
5. Click **Deploy** — live in ~2 minutes!

## Dataset

Based on the [House Rent Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset) from Kaggle.

Cities: Mumbai, Delhi, Bangalore, Kolkata, Chennai, Hyderabad
Records: 4,746 properties

## Author

**Viknesh C** — Junior Data Scientist  
[GitHub](https://github.com/cvviknesh) | [LinkedIn](https://linkedin.com/in/viknesh01)
