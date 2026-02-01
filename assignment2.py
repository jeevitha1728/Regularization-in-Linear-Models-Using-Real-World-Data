import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error


np.random.seed(50)

n_samples = 500

X = np.random.randn(n_samples, 6)

y = (
    5 * X[:, 0]
    + 3 * X[:, 1]
    - 2 * X[:, 2]
    + np.random.normal(0, 2, n_samples)
)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.25, random_state = 50 )

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

##multi linear regression
linear = LinearRegression()

linear.fit(X_train_scaled, y_train)

train_pred = linear.predict(X_train_scaled)

test_pred = linear.predict(X_test_scaled)

print("Train MSE:", mean_squared_error(y_train, train_pred))

print("Test  MSE:", mean_squared_error(y_test, test_pred))

alphas = [0.001, 0.01, 0.1, 1 , 1, 10, 100 , 200 , 300 , 400 ,500 ]

plt.figure(figsize = (8 , 5))

#ridge 

for alpha in alphas:
    ridge = Ridge(alpha = alpha)

    ridge.fit(X_train_scaled, y_train)

    train_mse = mean_squared_error( y_train, ridge.predict(X_train_scaled))
    
    test_mse = mean_squared_error( y_test, ridge.predict(X_test_scaled))

    plt.scatter(alpha, train_mse, color="blue")

    plt.scatter(alpha, test_mse, color="red")

print(f" Ridge coefficients {ridge.coef_}")
      
plt.xscale("log")
plt.title("Ridge")
plt.show()

#lassoo
plt.figure(figsize = (8 , 5))

for alpha in alphas:
        lasso = Lasso(alpha=alpha)

        lasso.fit(X_train_scaled, y_train)

        train_mse = mean_squared_error( y_train, lasso.predict(X_train_scaled) )

        test_mse = mean_squared_error(  y_test, lasso.predict(X_test_scaled)  )

        plt.scatter(alpha, train_mse, color="blue")
        
        plt.scatter(alpha, test_mse, color="red")

print(f" lasso coefficients {lasso.coef_}")

plt.xscale("log")
plt.title("Lasso")
plt.show()

# Elastic

plt.figure(figsize = (8 , 5))

plt.figure()

for alpha in alphas:
    enet = ElasticNet(alpha=alpha)
    
    enet.fit(X_train_scaled, y_train)

    train_mse = mean_squared_error( y_train, enet.predict(X_train_scaled) )
    
    test_mse = mean_squared_error( y_test, enet.predict(X_test_scaled) )

    plt.scatter(alpha, train_mse, color="blue")

    plt.scatter(alpha, test_mse, color="red")

plt.xscale("log")
plt.title("Elastic Net")
plt.show()
