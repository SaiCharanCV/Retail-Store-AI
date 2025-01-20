import os
import pickle
from tensorflow import keras
from keras.models import load_model
from PyQt5 import QtWidgets, QtGui

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class RetailStoreAI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Retail Store AI")
        self.setGeometry(100, 100, 600, 400)

        # Labels and input fields
        layout = QtWidgets.QVBoxLayout()

        layout.addWidget(QtWidgets.QLabel("Retail Store AI: Predict Demand and Competitor Pricing"))

        self.category_label = QtWidgets.QLabel("Product Category:")
        layout.addWidget(self.category_label)
        self.category_input = QtWidgets.QLineEdit(self)
        layout.addWidget(self.category_input)

        self.inventory_label = QtWidgets.QLabel("Inventory Level:")
        layout.addWidget(self.inventory_label)
        self.inventory_input = QtWidgets.QLineEdit(self)
        layout.addWidget(self.inventory_input)

        self.units_sold_label = QtWidgets.QLabel("Units Sold:")
        layout.addWidget(self.units_sold_label)
        self.units_sold_input = QtWidgets.QLineEdit(self)
        layout.addWidget(self.units_sold_input)

        self.units_ordered_label = QtWidgets.QLabel("Units Ordered:")
        layout.addWidget(self.units_ordered_label)
        self.units_ordered_input = QtWidgets.QLineEdit(self)
        layout.addWidget(self.units_ordered_input)

        self.price_label = QtWidgets.QLabel("Price of the Product:")
        layout.addWidget(self.price_label)
        self.price_input = QtWidgets.QLineEdit(self)
        layout.addWidget(self.price_input)

        self.discount_label = QtWidgets.QLabel("Discount Offered:")
        layout.addWidget(self.discount_label)
        self.discount_input = QtWidgets.QLineEdit(self)
        layout.addWidget(self.discount_input)

        self.holiday_label = QtWidgets.QLabel("Any Holiday or Promotion (yes/no):")
        layout.addWidget(self.holiday_label)
        self.holiday_input = QtWidgets.QLineEdit(self)
        layout.addWidget(self.holiday_input)

        # Predict button
        self.predict_button = QtWidgets.QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict)
        layout.addWidget(self.predict_button)

        # Output area
        self.output = QtWidgets.QTextEdit(self)
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

        self.setLayout(layout)

    def predict(self):
        try:
            # Load models and scalers
            with open("E:\Python Projects\Retail Store AI\Retail_Store_Demand_Predictor_with_le.pkl", 'rb') as f1, \
                 open("E:\Python Projects\Retail Store AI\Retail_Store_Competitor_Price_Predictor_with_le.pkl", 'rb') as f2:
                load_demand_predictor = pickle.load(f1)
                load_competitor_price_predictor = pickle.load(f2)

            demand_scaler = load_demand_predictor['scaler']
            demand_predictor_le = load_demand_predictor['Label Encoder']
            demand_forecast_max = load_demand_predictor['Demand Forecast Max']

            competitive_scaler = load_competitor_price_predictor['scaler competitor price']
            model_demand_predictor = load_model("E:\Python Projects\Retail Store AI\model_demand_predictor.h5")
            model_competitive_predictor = load_model("E:\Python Projects\Retail Store AI\model_competitive_price_predictor.h5")
            competitor_price_max = load_competitor_price_predictor['Competitor Price Max']

            # Get inputs
            category = self.category_input.text()
            inventory = float(self.inventory_input.text())
            units_sold = float(self.units_sold_input.text())
            units_ordered = float(self.units_ordered_input.text())
            price = float(self.price_input.text())
            discount = float(self.discount_input.text())
            holiday = self.holiday_input.text().lower()

            holiday = 1 if holiday == 'yes' else 0
            labeled_category = demand_predictor_le.transform([category])[0]

            # Predict demand
            demand_input = [labeled_category, inventory, units_sold, units_ordered, price, discount, holiday]
            scaled_demand_input = demand_scaler.transform([demand_input])
            predicted_demand = model_demand_predictor.predict(scaled_demand_input)[0][0] * demand_forecast_max

            # Predict competitor price
            cp_input = [labeled_category, inventory, units_sold, units_ordered, predicted_demand, price, discount, holiday]
            scaled_cp_input = competitive_scaler.transform([cp_input])
            predicted_cp = model_competitive_predictor.predict(scaled_cp_input)[0][0] * competitor_price_max

            # Display results
            self.output.setText(f"Predicted Demand: {predicted_demand:.2f}\nPredicted Competitor Price: {predicted_cp:.2f}")
        except Exception as e:
            self.output.setText(f"Error: {str(e)}")


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = RetailStoreAI()
    window.show()
    app.exec_()
