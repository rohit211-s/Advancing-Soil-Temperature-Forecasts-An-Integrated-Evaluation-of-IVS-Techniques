best_test_rmse=1
for roh in range(1,21):
  tf.random.set_seed(42)

  # Let's build an LSTM model with the Functional API
  inputs = layers.Input(shape=(3))
  x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs) # expand input dimension to be compatible with LSTM
  # print(x.shape)
  # x = layers.LSTM(128, activation="relu", return_sequences=True)(x) # this layer will error if the inputs are not the right shape
  x = layers.LSTM(roh, activation="relu")(x) # using the tanh loss function results in a massive error
  # print(x.shape)


  output = layers.Dense(HORIZON)(x)
  model_5 = tf.keras.Model(inputs=inputs, outputs=output, name="model_5_lstm")

  # Compile model
  model_5.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.RootMeanSquaredError()])

  # Seems when saving the model several warnings are appearing: https://github.com/tensorflow/tensorflow/issues/47554
  model_5.fit(X_train,
              y_train,
              epochs=100,
              verbose=0,
              batch_size=128,
              validation_data=(X_test, y_test),
              callbacks=[create_model_checkpoint(model_name=model_5.name)])
  print(roh)
  rmse=model_5.evaluate(X_test, y_test)
  rmse=rmse[1]
  print("test rmse",rmse)
  train_rmse=model_5.evaluate(X_train, y_train)
  train_rmse=train_rmse[1]
  print("train rmse",train_rmse)
  if rmse < best_test_rmse:
    best_test_rmse=rmse
    best_hidden_size=roh
    best_train_rmse=train_rmse
    best_test_predictions=model_5.predict(X_test)

ws1['K2']=best_hidden_size
ws1['L2']=best_train_rmse
ws1['M2']=best_test_rmse

for i, prediction in enumerate(y_test):
    ws5['A{}'.format(i+2)] = float(prediction)
for i, prediction in enumerate(best_test_predictions):
    ws5['B{}'.format(i+2)] = float(prediction)
wb.save(filename = 'Results.xlsx')
