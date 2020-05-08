import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense


def PCA_(X, n_c):
    pca = PCA()
    pca.fit(X)
    pcavar = pca.explained_variance_ratio_
    cum_exp_var = np.cumsum(pcavar)
    plt.plot(cum_exp_var)
    plt.xlabel('Principal components')
    plt.ylabel('Cumulative explained variance')
    plt.savefig('PCA_cumulative_explained_variance')

    pca = PCA(n_components=n_c)
    pca.fit(X)
    y_latent_PCA = pca.transform(X)
    y_pred_PCA = pca.inverse_transform(y_latent_PCA)
    return y_pred_PCA

def plots(method,index_train, nasdaq100_train, tracked_index_insample, index_test, nasdaq100_test, tracked_index_outofsample):
    # Plot tracked index in sample
    plt.figure()
    plt.plot(index_train, label='Nasdaq Composite')
    plt.plot(nasdaq100_train, label='Nasdaq 100')
    plt.plot(tracked_index_insample, label='Tracked'+method)
    plt.legend()
    plt.xlabel('Time (days)')
    plt.ylabel('Normalized price')
    plt.ylim((0, 1))
    plt.savefig(method+'in sample Normalized price')

    # Plot tracked index (out-of-sample)
    plt.figure()
    plt.plot(index_test, label='Nasdaq Composite')
    plt.plot(nasdaq100_test, label='Nasdaq 100')
    plt.plot(tracked_index_outofsample, label='Tracked'+method)
    plt.legend()
    plt.xlabel('Time (days)')
    plt.ylabel('Normalized price')
    plt.ylim((0, 1))
    plt.savefig(method+'out sample Normalized price')

    # Correlation coefficient
    corr_train = np.corrcoef(index_train.squeeze(), tracked_index_insample)[0, 1]
    corr_index = np.corrcoef(index_train.squeeze(), nasdaq100_train.squeeze())[0, 1]
    print(method,'Correlation coefficient of portfolio(in-sample): %.8f' % corr_train)
    print(method,'Correlation coefficient of nasdaq100(in-sample): %.8f' % corr_index)
    # Correlation coefficient (out-of-sample)
    corr_test = np.corrcoef(index_test.squeeze(), tracked_index_outofsample)[0, 1]
    corr_index_test = np.corrcoef(index_test.squeeze(), nasdaq100_test.squeeze())[0, 1]
    print(method,'Correlation coefficient of portfolio(out-sample): %.8f' % corr_test)
    print(method,'Correlation coefficient of nasdaq100(out-sample): %.8f' % corr_index_test)

# load data
stocks = pd.read_pickle("nasdaq_stocks_price.pickle")
index = pd.read_pickle("nasdaq_composite_index.pickle")
nasdaq100 = pd.read_pickle("nasdaq100_index.pickle")
# Remove the missing the data from the dataframe
stocks= stocks.dropna(axis=1)
assets_names = stocks.columns.values
# Normalize data
scaler = MinMaxScaler([0.1, 0.9])
stocks = scaler.fit_transform(stocks)
scaler_index = MinMaxScaler([0.1, 0.9])
index = scaler_index.fit_transform(index)
nasdaq100 = scaler_index.fit_transform(nasdaq100)
X_train, X_test = stocks[:1000, :], stocks[1000:, :]
index_train, index_test = index[:1000, :], index[1000:, :]
nasdaq100_train, nasdaq100_test = nasdaq100[:1000, :], nasdaq100[1000:, :]
print("Stocks data (time series): Trianing set", X_train.shape,"; Test set:", X_test.shape)
print("Index data (time series): Trianing set",index_train.shape, "; Test set:",index_test.shape)
# Generate corrupted series by adding gaussian noise
noise_factor = 0.05
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)
# Clip corrupted data
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)


### Training autoencoders
epochs = 20
batch_size = 1
n_inputs = X_train.shape[1]
# Create the model
input = Input(shape=(n_inputs,))
encoded = Dense(200, input_shape=(n_inputs,), activation='relu', activity_regularizer=regularizers.l1(10e-8))(input)
encoded = Dense(50, activation='relu',activity_regularizer=regularizers.l1(10e-8))(encoded)
decoded = Dense(200, activation='relu', activity_regularizer=regularizers.l1(10e-8))(encoded)
decoded = Dense(n_inputs, activation='sigmoid')(decoded)
model = Model(input, decoded)
model.summary()
model.compile(loss='mse', optimizer=Adam(lr=1e-4))
history = model.fit(X_train_noisy, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=2)

# Visualize training loss
plt.figure()
plt.plot(history.history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('Loss')

# Evaluate model
score_train = model.evaluate(X_train_noisy, X_train, batch_size=batch_size)
score_test = model.evaluate(X_test_noisy, X_test, batch_size=batch_size)
print('Training MSE: %.8f' %score_train)
print('Test MSE: %.8f' %score_test)

# Obtain reconstruction of the stocks
X_train_pred = model.predict(X_train_noisy)
# Reconstruction error
error = np.mean(np.abs(X_train - X_train_pred)**2, axis=0)

# Identify stocks
n = 20
# Sort stocks
ind = np.argsort(error)
sort_error = error[ind]
sort_assets_names = assets_names[ind]
# Barplot
plt.figure()
plt.barh(2*np.arange(len(error[:n])), error[ind[:n]], tick_label=assets_names[ind[:n]])
plt.xlabel('MSE')
plt.savefig('AE_Top20 composite')
portfolio_train = X_train_pred[:, ind[:n]]
# Create portfolio
tracked_AE_insample = np.mean(portfolio_train, axis=1)
# out of sample
portfolio_test = X_test[:, ind[:n]]
tracked_AE_outofsample = np.mean(portfolio_test, axis=1)
plots(str('AE'), index_train, nasdaq100_train, tracked_AE_insample, index_test, nasdaq100_test, tracked_AE_outofsample)

### PCA
pca_train_pred= PCA_(X_train, 50)
# Reconstruction error
pca_error = np.mean(np.abs(X_train - pca_train_pred)**2, axis=0)

# Sort stocks
pca_ind = np.argsort(pca_error)
sort_error = pca_error[pca_ind]
sort_assets_names = assets_names[pca_ind]
# Barplot
plt.figure()
plt.barh(2*np.arange(len(pca_error[:n])), pca_error[pca_ind[:n]], tick_label=assets_names[pca_ind[:n]])
plt.xlabel('MSE')
plt.savefig('PCA_Top20 composite')

# Create portfolio
portfolio_train_PCA = pca_train_pred[:, pca_ind[:n]]
tracked_PCA_insample = np.mean(portfolio_train_PCA, axis=1)
# OUT OF SAMPLE
portfolio_test_PCA = X_test[:, pca_ind[:n]]
tracked_PCA_outofsample = np.mean(portfolio_test_PCA, axis=1)
plots(str('PCA'), index_train, nasdaq100_train, tracked_PCA_insample, index_test, nasdaq100_test, tracked_PCA_outofsample)

# Plot series
for stk in range(10):
    plt.figure()
    plt.plot(X_train[: , pca_ind[stk]], label='X')
    plt.plot(pca_train_pred[: , pca_ind[stk]], label='PCA reconstruction')
    plt.plot(X_train_pred[: , pca_ind[stk]], label='AE reconstruction')
    plt.title(assets_names[pca_ind[stk]])
    plt.legend()
    plt.savefig(assets_names[pca_ind[stk]]+'.png', bbox_inches='tight')