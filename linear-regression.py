# Bước 1: Đọc và xử lý dữ liệu

import pandas as pd
import seaborn as sns
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # Chia tách dữ liệu
from sklearn.preprocessing import StandardScaler      # Đưa dữ liệu về dạng phân phối chuẩn
from sklearn.linear_model import LinearRegression     # Mô hình hồi quy tuyến tính
from sklearn import metrics
from sklearn.metrics import mean_squared_error

# Đọc file
file_path = 'du_lieu_co_phieu.csv'
df = pd.read_csv(file_path)

#Xóa các cột không dùng
df = df.drop(columns=["Date", "Dividends", "Stock Splits"])

print(df.head())

print(df.shape)

print(df.info())

print(df.isnull().sum())

# Khai báo các biến

O = df["Open"].values
H = df["High"].values
L = df["Low"].values
C = df["Close"].values
V = df["Volume"].values

# Vẽ đồ thị thể hiện mối quan hệ
plt.figure(figsize=(15,8))

plt.subplot(2,2,1)
plt.scatter(O, C, color='coral', alpha=0.5)
plt.title("Giá mở cửa và giá đóng cửa", fontsize=12, color="darkblue")
plt.xlabel("Giá mở cửa", fontsize=12)
plt.ylabel("Giá đóng cửa", fontsize=12)

plt.subplot(2,2,2)
plt.scatter(H, C, color='lime', alpha=0.5)
plt.title("Giá trần và giá đóng cửa", fontsize=12, color="darkblue")
plt.xlabel("Giá trần", fontsize=12)
plt.ylabel("Giá đóng cửa", fontsize=12)

plt.subplot(2,2,3)
plt.scatter(L, C, color='steelblue', alpha=0.5)
plt.title("Giá sàn và giá đóng cửa", fontsize=12, color="darkblue")
plt.xlabel("Giá sàn", fontsize=12)
plt.ylabel("Giá đóng cửa", fontsize=12)

plt.subplot(2,2,4)
plt.scatter(V, C, color='orchid', alpha=0.5)
plt.title("Khối lượng giao dịch và giá đóng cửa", fontsize=12, color="darkblue")
plt.xlabel("Khối lượng giao dịch", fontsize=12)
plt.ylabel("Giá đóng cửa", fontsize=12)
plt.tight_layout()

# bỏ cột giá đóng cửa từ file dữ liệệu đầu vào

x = df.drop("Close", axis=1).values
print(x)

# tách giá đóng cửa ra một mảng mới
y = df["Close"].values
y = y.reshape(-1, 1)
print(y)

# Bước 2: xây dựng mô hình
# Chia bộ dữ liệu thành hai tập: train & test (theo tỉ lệ 70% & 30%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Tính tỉ lệ phần trăm của tập train và test so với dữ liệu gốc
Training_to_original_ratio = round(x_train.shape[0] / (df.shape[0]), 2) * 100
Testing_to_original_ratio = round(x_test.shape[0] / (df.shape[0]), 2) * 100

# In kết quả
print('Training: {}%; Testing: {}%'.format(Training_to_original_ratio, Testing_to_original_ratio))
print(list(zip(['Training set', 'Testing set'], [Training_to_original_ratio, Testing_to_original_ratio])))

# Chuẩn hóa tập train và test về dạng phân phối chuẩn bằng hàm StandardScaler
std_scale = StandardScaler().fit(x_train)  # Khớp scaler với dữ liệu train
x_train_scaled = std_scale.transform(x_train)  # Áp dụng scaler cho tập train
x_test_scaled = std_scale.transform(x_test)    # Áp dụng cùng scaler cho tập test

# Khai báo mô hình hồi quy tuyến tính
Linear_reg = LinearRegression()

# Bước 3: Huấn luyện và xác định độ chính xác của mô hình

# Đào tạo mô hình
reg_scaled = Linear_reg.fit(x_train_scaled, y_train)
y_train_scaled_fit = reg_scaled.predict(x_train_scaled)

# Độ phù hợp của mô hình (R-squared)
print("R-squared for training dataset: {}".format(
    np.round(reg_scaled.score(x_train_scaled, y_train), 2)))

# Lỗi trung bình bình phương (RMSE)
print("Root mean square error: {}".format(
    np.round(np.sqrt(mean_squared_error(y_train, y_train_scaled_fit)), 2)))

# Hệ số các biến
coefficients = reg_scaled.coef_
feature = list(df.drop("Close", axis=1).columns)

print("\nHệ số của các biến độc lập:")
print(dict(zip(feature, coefficients[0])))

print("\nHệ số chặn: {}".format(np.round(reg_scaled.intercept_, 3)))

# Bước 4: Sử dụng mô hình

# Dự báo kết quả bằng hàm predict
pred = reg_scaled.predict(x_test_scaled)
print("Giá dự đoán:\n", pred)

# vẽ biểu đồ thể hiện kết quả
data = {"prediction": pred, "observed": y_test}
test = pd.DataFrame(pred, columns=["Prediction"])
test["Observed"] = y_test

# Tính đường làm mịn LOWESS (phân tán có trọng số cục bộ)
lowess = sm.nonparametric.lowess
z = lowess(pred.flatten(), y_test.flatten())

# Biểu đồ so sánh giá dự đoán và giá thực tế
test.plot(figsize=[12,12], x="Prediction", y="Observed", kind="scatter", color='red')
plt.title("Dự đoán giá cổ phiếu với tập test", fontsize=12, color="indigo")
plt.xlabel("Giá cổ phiếu dự đoán", fontsize=12)
plt.ylabel("Giá cổ phiếu thực tế", fontsize=12)
plt.plot(z[:,0], z[:,1], color="midnightblue", lw=3)

# Độ phù hợp với tập test
print("R-squared for test dataset: {}".format(np.round(reg_scaled.score(x_test_scaled, y_test), 2)))

# Lỗi trung bình bình phương RMSE
print("Root mean square error: {}".format(np.round(np.sqrt(mean_squared_error(y_test, pred)), 2)))

test1 = [[211, 213, 210, 23889458]]  # Dữ liệu đầu vào cần dự đoán

# Chuẩn hóa test1 bằng cùng scaler đã fit từ x_train
test1_scaled = std_scale.transform(test1)

# Dự đoán giá Close
prediction_test1 = reg_scaled.predict(test1_scaled)

print("Giá cổ phiếu dự đoán là:", np.round(prediction_test1[0][0], 2))


plt.show()