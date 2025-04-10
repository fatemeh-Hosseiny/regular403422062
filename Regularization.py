import numpy as np
import matplotlib.pyplot as plt
# No longer importing sklearn modules

# تولید داده
np.random.seed(0) # باعث تولید داده های یکسان در هر تکرار میشه
n_samples = 10
sigma_squared = 0.1
X = np.sort(np.random.rand(n_samples)) 
y_true = np.sin(np.pi * X) 
noise = np.sqrt(sigma_squared) * np.random.randn(n_samples) # Gaussian noise
y = y_true + noise # داده‌های آموزشی نهایی (با نویز)

print("Input Data X:", X)
print("Target Data y:", y)

# --- تابع  برای تبدیل ویژگی‌ها به چندجمله‌ای---
def manual_polynomial_features(X_in, degree):
    # یک ورودی یک بعدی است x_in
    X_in = X_in.reshape(-1, 1) # یک ستون است
    n_samples = X_in.shape[0]
    
    
    X_poly = np.ones((n_samples, 1))
    
    # اضافه کردن ستون‌های x^1, x^2, ..., x^degree
    for d in range(1, degree + 1):
        X_poly = np.hstack((X_poly, np.power(X_in, d)))
        
    return X_poly

# تبدیل به ویژگی تابع چند جمله ای درجه 9
degree = 9
X_poly = manual_polynomial_features(X, degree)
print("Shape of Polynomial Features Matrix:", X_poly.shape)


# --- L2 Regularization ---
def sgd_ridge(X, y, learning_rate=0.01, lambda_reg=0.01, n_epochs=5000):
    """ SGD implementation  """
    n_samples, n_features = X.shape
    w = np.zeros(n_features) # مقداردهی اولیه وزن‌ها

    losses = [] # برای ذخیره تابع زیان

    for epoch in range(n_epochs):
        total_epoch_loss = 0
        indices = np.random.permutation(n_samples) # تصادفی کردن ترتیب نمونه‌ها

        for i in indices:
            xi = X[i]
            yi = y[i]
            
            prediction = np.dot(xi, w)
            error = yi - prediction
            
            # محاسبه گرادیان
            gradient = -error * xi + lambda_reg * w
            
            # بروزرسانی وزن ها
            w = w - learning_rate * gradient
            
            # محاسبه زیان برای این نمونه

            loss = 0.5 * error**2 + 0.5 * lambda_reg * np.dot(w, w)
            losses.append(loss)
            total_epoch_loss += loss

    return w, losses

# ---L1 Regularization---
def sgd_lasso(X, y, learning_rate=0.01, lambda_reg=0.001, n_epochs=5000):
    """ SGD implementation  """
    n_samples, n_features = X.shape
    w = np.zeros(n_features) # مقدار دهی وزن ها
    losses = [] # برای ذخیره تابع زیان

    for epoch in range(n_epochs):
        total_epoch_loss = 0
        indices = np.random.permutation(n_samples) # تصادفی کردن ترتیب نمونه‌ها

        for i in indices:
            xi = X[i]
            yi = y[i]
            
            prediction = np.dot(xi, w)
            error = yi - prediction
            
            # محاسبه گرادیان زیرین

            subgradient = -error * xi + lambda_reg * np.sign(w)
            
            # بروزرسانی وزن ها

            w = w - learning_rate * subgradient

            # محاسبه زیان برای این نمونه

            loss = 0.5 * error**2 + lambda_reg * np.sum(np.abs(w))
            losses.append(loss)
            total_epoch_loss += loss
            
    return w, losses


# ---  Run Models and Predict ---
# Run SGD for Ridge
w_ridge_sgd, losses_ridge = sgd_ridge(X_poly, y, learning_rate=0.01, lambda_reg=0.01, n_epochs=5000)

# Run SGD for Lasso
w_lasso_sgd, losses_lasso = sgd_lasso(X_poly, y, learning_rate=0.01, lambda_reg=0.001, n_epochs=8000)

print("\nFinal Weights Ridge (SGD):", w_ridge_sgd)
print("Final Weights Lasso (SGD):", w_lasso_sgd)
# دیگر وزن‌های بیزی وجود ندارد

# --- Plot Results ---
# ایجاد نقاط برای رسم منحنی‌های هموار
X_plot = np.linspace(0, 1, 100) # آرایه یک بعدی
# تبدیل نقاط جدید با تابع
X_plot_poly = manual_polynomial_features(X_plot, degree)

# پیش‌بینی با مدل‌های یادگرفته شده روی نقاط جدید
y_pred_ridge_sgd_plot = X_plot_poly @ w_ridge_sgd
y_pred_lasso_sgd_plot = X_plot_poly @ w_lasso_sgd
# دیگر پیش‌بینی بیزی وجود ندارد
y_true_plot = np.sin(np.pi * X_plot) # True function on new points

plt.figure(figsize=(12, 7))
plt.scatter(X, y, color='blue', marker='o', s=60, label='Training Data (N=10, $\sigma^2$=0.1)')
plt.plot(X_plot, y_true_plot, color='green', linewidth=2, label='True Function $sin(\pi x)$')
plt.plot(X_plot, y_pred_ridge_sgd_plot, color='red', linestyle='--', linewidth=2, label='Ridge Fit (SGD, $\lambda$=0.01)')
plt.plot(X_plot, y_pred_lasso_sgd_plot, color='purple', linestyle='-.', linewidth=2, label='Lasso Fit (SGD, $\lambda$=0.001)')
# دیگر پلات بیزی وجود ندارد

plt.xlabel('x')
plt.ylabel('y')
plt.title('Degree 9 Polynomial Fit with SGD (No sklearn)')
plt.legend(loc='best')
plt.grid(True)
plt.ylim(-1.5, 1.5) # Adjust y-axis limits for better visualization
plt.show()

#  پلات تابع زیان برای SGD

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses_ridge)
plt.title('Ridge Loss (SGD) per Update')
plt.xlabel('Update Number')
plt.ylabel('Instantaneous Loss')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(losses_lasso)
plt.title('Lasso Loss (SGD) per Update')
plt.xlabel('Update Number')
plt.ylabel('Instantaneous Loss')
plt.grid(True)

plt.tight_layout() # Adjust subplot parameters for a tight layout
plt.show()