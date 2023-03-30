import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load and store the data
data = data = pd.read_csv("C:\\AppSales.csv")

#CLEAN AND OPTIMIZE
# Drop columns that are not needed.
data = data.drop(["Size", "Type", "Content Rating", "Last Updated", "Current Ver", "Android Ver", "Reviews"], axis=1)

# Convert the "Price" column to float and remove the dollar sign
data["Price"] = data["Price"].apply(lambda x: str(x).replace("$", "")).astype(float)
data["Installs"] = data["Installs"].apply(lambda x: str(x).replace("+", ""))
data["Installs"] = data["Installs"].apply(lambda x: str(x).replace(",", "")).astype(float)

# Filter out rows where the price is zero because we don't need them
data = data[data["Price"] > 0]

# Calculate the earnings and add a new column to the table
data["Earnings"] = data["Installs"] * data["Price"]

# Group the data by category and sum the installs and earnings
grouped_data = data.groupby("Category").agg({"Installs": "sum", "Earnings": "sum"})

# Calculate the earnings growth rate, can get this data from external api from business website or something.
growth_rate = 1.1

# Predict the earnings for the next year
X = grouped_data[["Installs"]]
y = grouped_data["Earnings"]

model = LinearRegression().fit(X.values, y)

next_year_installs = X.max()[0] * growth_rate
next_year_earnings = model.predict([[next_year_installs]])[0]


@app.route('/earnings-prediction', methods=['POST'])
def get_earnings_prediction():
    # Get user input for the game category and convert it to uppercase
    category = request.json.get('category', '').upper()

    # Get the earnings prediction for the specified category
    if category in grouped_data.index:
        category_installs = grouped_data.loc[category]["Installs"]
        category_earnings = grouped_data.loc[category]["Earnings"]
        predicted_earnings = category_earnings * (next_year_earnings / category_earnings)

        # Y-axis values
        x = [category_earnings, predicted_earnings]
        y = [2023, 2024]

        # Function to plot
        plt.plot(x, y)
        plt.xlabel("Earning")
        plt.ylabel("Year & Month")

        # Function add a legend
        plt.legend(["blue", "green"], loc="lower right")

        # Function to show the plot
        plt.show()
        response = {'predicted_earnings': predicted_earnings}
        return jsonify(response), 200
    else:
        response = {'error': f"{category} not found in the data"}
        return jsonify(response), 400

if __name__ == '__main__':
    app.run(debug=True)