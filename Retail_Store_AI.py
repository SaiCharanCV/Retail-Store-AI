import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
import pickle
from tensorflow import keras
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

print("Retail Store AI Loading...")
with open("Retail_Store_Demand_Predictor_with_le.pkl", 'rb') as f1, open("Retail_Store_Competitor_Price_Predictor_with_le.pkl", 'rb') as f2:
    load_demand_predictor = pickle.load(f1)
    load_competitor_price_predictor = pickle.load(f2)

demand_scaler = load_demand_predictor['scaler']
demand_predictor_le = load_demand_predictor['Label Encoder']
demand_forecast_max = load_demand_predictor['Demand Forecast Max']
competitive_scaler = load_competitor_price_predictor['scaler competitor price']
model_demand_predictor = load_model("model_demand_predictor.h5")
model_competitive_predictor = load_model("model_competitive_price_predictor.h5")
competitor_price_max = load_competitor_price_predictor['Competitor Price Max']
print(f"""
      Hello. Im Retail Store AI. I can help you predict demand for products tomorrow and can also predict your competitor's Price for the product""")

demand_product_category = input("Product Category: ")
labeled_product_category = demand_predictor_le.transform([demand_product_category])
inventory_level = float(input("Inventory Level: "))
units_sold = float(input("Units Sold: "))
units_ordered = float(input("Units Ordered: "))
price = float(input("Price of the product: "))
discount = float(input("Discount you offer on the product: "))
holiday_promotion = input("Any Holiday or Promotion today (yes/no): ").lower()
holiday_promotion = 1 if holiday_promotion == 'yes' else 0
labeled_product_category = labeled_product_category[0]
demand_input = [labeled_product_category, inventory_level, units_sold, units_ordered, price, discount, holiday_promotion]
scaled_demand_input = demand_scaler.transform([demand_input])
model_predicted_demand = model_demand_predictor.predict(scaled_demand_input)
model_predicted_demand = model_predicted_demand[0][0] * demand_forecast_max
cp_input = [labeled_product_category, inventory_level, units_sold, units_ordered, model_predicted_demand, price, discount, holiday_promotion]
scaled_cp_input = competitive_scaler.transform([cp_input])
model_predicted_cp = model_competitive_predictor.predict(scaled_cp_input)
predicted_cp = model_predicted_cp[0][0] * competitor_price_max



if price > predicted_cp:
    if model_predicted_demand < 100:
        model_suggestion = 'Competitor cost is cheaper than yours and your forecasted demand is also low. Try Reducing the price lesser than competitor if that is feasible with you'
    else:
        model_suggestion = 'Competitor price is less than your price but forecasted demand is good. Current price is fine with the forecasted demand'
else:
    model_suggestion = 'Your Price is less than Competitor Price. If your demand is low, Try reducing the price if that is possible'
print(f"Model Predicted Competitor Price: {predicted_cp:.2f}\nModel Predicted Demand: {model_predicted_demand:.2f}\nModel's Suggestion: {model_suggestion}")
