import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error as mse
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from mpl_toolkits.mplot3d import Axes3D as plt3d
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture



def extract_data(file_path, n_samples):
    file_names = [f for f in os.listdir(file_path)]
    data = [[], [], [], []]

    # Extracts the x,y,z data from each .csv file for each data time
    for file_name in file_names:
        file = pd.read_csv(file_path + "\\" + file_name, sep=";")
        for i in range(4):
            data[i] += [[float(s.replace(",", ".")) for s in [x, y, z]] for x, y, z in
                        zip(file.iloc[1:, ((i*3)+2)], file.iloc[1:, ((i*3)+3)], file.iloc[1:, ((i*3)+4)])]
    np.random.shuffle(data)
    data = np.array(data)[:n_samples, :]
    return data



def plot_data(data):
    #Contains newly formatted data to display via 'pyplot' as:
    # 'num categories (accel, gyro, magnet)' x 'num dimensions (x, y, z)' x 'num data points over all files'
    xyz_data = [[], [], []]
    for i in range(3):
        for j in range(len(data)):
            xyz_data[j].append(pd.DataFrame(data[j]).iloc[:, i].tolist())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #Points range to plot
    p_range = [0, 1000]
    for i, c in enumerate(('r', 'g', 'b')):
        ax.scatter(xyz_data[i][0][p_range[0]:p_range[1]], xyz_data[i][1][p_range[0]:p_range[1]],
                   xyz_data[i][2][p_range[0]:p_range[1]], c=c)
    plt.show()



def cluster_analysis(data, n_clusters, c_choice):

    #Shuffles the samples and selects only a subset of them (otherwise, plotting over 300k samples!)
    if c_choice == "kmc":
        accel_pred = KMeans(n_clusters=n_clusters).fit_predict(data)
    elif c_choice == "gmm":
        accel_pred = GaussianMixture(n_components=n_clusters).fit_predict(data)
    elif c_choice == "agc":
        accel_pred = AgglomerativeClustering(n_clusters=n_clusters, linkage='single').fit_predict(data)
    else:
        return
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=accel_pred)
    plt.show()



def supervised_analysis(data, r_choice):

    x = np.concatenate((data[1], data[2], data[3]), axis=1)
    y = data[0]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    if r_choice == "nn":
        model = Sequential()
        model.add(Dense(units=100, activation='relu', input_shape=(9,)))
        model.add(Dense(units=50, activation='relu'))
        model.add(Dense(units=25, activation='relu'))
        model.add(Dense(units=3, activation='linear'))
        model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mse'])
        model.fit(x=x_train, y=y_train, batch_size=128, epochs=10, verbose=0)
        error = model.evaluate(x_test, y_test, verbose=0)[1]
        print("\nNeural Net MSE:", error)

    elif r_choice == "lr":
        lr = LinearRegression(normalize=True).fit(x_train, y_train)
        y_predict = lr.predict(x_test)
        error = mse(y_true=y_test, y_pred=y_predict)
        print("\nLinear Regression MSE:", error)

    elif r_choice == "knnr":
        knnr = KNeighborsRegressor(n_neighbors=5, weights='distance').fit(x_train, y_train)
        y_predict = knnr.predict(x_test)
        error = mse(y_true=y_test, y_pred=y_predict)
        print("\nK-nearest Neighbors Regression MSE:", error)

    elif r_choice == "rid":
        rid = Ridge().fit(x_train, y_train)
        y_predict = rid.predict(x_test)
        error = mse(y_true=y_test, y_pred=y_predict)
        print("\nRidge Regression MSE:", error)

    elif r_choice == "all":
        supervised_analysis(data, "nn")
        supervised_analysis(data, "lr")
        supervised_analysis(data, "knnr")
        supervised_analysis(data, "rid")



#Change as needed to point to dataset; system dependent
pend_file_path = "..\\RepoIMU\\RepoIMU\\Pendulum"

#'pend_data' is in dimensions 'num types (accel, gyro, magnet)' x 'num samples' x 'num dimensions (x, y, z)'
pend_data = extract_data(pend_file_path, n_samples=10000)

plot_data(pend_data)
cluster_analysis(pend_data[3], n_clusters=4, c_choice="agc")
supervised_analysis(pend_data, r_choice="all")