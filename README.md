Name :Kasireddy.Abhiram reddy
company name: codtech solution
ID:CT08DS1502
duration:june 1st to 30th june
mentor: G.Sravani
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
data = pd.read_csv('data.csv')

# Display basic information
print(data.info())
print(data.describe())

# Handling missing values
imputer = SimpleImputer(strategy='mean')
data['column_with_missing_values'] = imputer.fit_transform(data[['column_with_missing_values']])

# Encoding categorical variables
categorical_features = ['categorical_column1', 'categorical_column2']
one_hot_encoder = OneHotEncoder()
data_encoded = pd.get_dummies(data, columns=categorical_features)

# Feature scaling
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_encoded)

# Splitting the data
X = data_scaled.drop('target_column', axis=1)
y = data_scaled['target_column']
X_train, X_test, y_train, y_test = train_test_split(X, y, t
