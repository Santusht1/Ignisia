import pandas as pd
import numpy as np
import datetime
import os

os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

np.random.seed(42)
dates = pd.date_range(end=datetime.date.today(), periods=90)
products = ['Product A', 'Product B', 'Product C']

data = []
for date in dates:
    weekend_multiplier = 1.5 if date.dayofweek >= 5 else 1.0
    for product in products:
        price = 20.0 if product == 'Product A' else (50.0 if product == 'Product B' else 10.0)
        base_qty = 10 if product == 'Product A' else (5 if product == 'Product B' else 30)
        trend = (date - dates[0]).days * 0.1
        qty = int(max(0, (base_qty + trend + np.random.normal(0, 5)) * weekend_multiplier))
        data.append({
            'Date': date.strftime('%Y-%m-%d'),
            'Product': product,
            'Price': price,
            'Quantity': qty
        })

df = pd.DataFrame(data)
df.to_csv(os.path.join(os.path.dirname(__file__), 'sample_sales.csv'), index=False)
print("sample_sales.csv generated.")
