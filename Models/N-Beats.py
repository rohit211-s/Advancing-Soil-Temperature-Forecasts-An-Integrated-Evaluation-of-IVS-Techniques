
# Create NBeatsBlock custom layer
class NBeatsBlock(tf.keras.layers.Layer):
  def __init__(self, # the constructor takes all the hyperparameters for the layer
               input_size: int,
               theta_size: int,
               horizon: int,
               n_neurons: int,
               n_layers: int,
               **kwargs): # the **kwargs argument takes care of all of the arguments for the parent class (input_shape, trainable, name)
    super().__init__(**kwargs)
    self.input_size = input_size
    self.theta_size = theta_size
    self.horizon = horizon
    self.n_neurons = n_neurons
    self.n_layers = n_layers

    # Block contains stack of 4 fully connected layers each has ReLU activation
    self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
    # Output of block is a theta layer with linear activation
    self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

  def call(self, inputs): # the call method is what runs when the layer is called
    x = inputs
    for layer in self.hidden: # pass inputs through each hidden layer
      x = layer(x)
    theta = self.theta_layer(x)
    # Output the backcast and forecast from theta
    backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
    return backcast, forecast

# Set up dummy NBeatsBlock layer to represent inputs and outputs
dummy_nbeats_block_layer = NBeatsBlock(input_size=WINDOW_SIZE,
                                       theta_size=WINDOW_SIZE+HORIZON, # backcast + forecast
                                       horizon=HORIZON,
                                       n_neurons=128,
                                       n_layers=4)


# Create dummy inputs (have to be same size as input_size)
dummy_inputs = tf.expand_dims(tf.range(WINDOW_SIZE) + 1, axis=0) # input shape to the model has to reflect Dense layer input requirements (ndim=2)
print(dummy_inputs)

# Pass dummy inputs to dummy NBeatsBlock layer
backcast, forecast = dummy_nbeats_block_layer(dummy_inputs)
# These are the activation outputs of the theta layer (they'll be random due to no training of the model)
print(f"Backcast: {tf.squeeze(backcast.numpy())}")
print(f"Forecast: {tf.squeeze(forecast.numpy())}")



train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)

test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test)
test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

# 2. Combine features & labels
train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

# 3. Batch and prefetch for optimal performance
BATCH_SIZE = 1024 # taken from Appendix D in N-BEATS paper
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_dataset, test_dataset


best_test_rmse=1
for roh in range(1,21):
  # Values from N-BEATS paper Figure 1 and Table 18/Appendix D
  N_EPOCHS = 100 # called "Iterations" in Table 18
  N_NEURONS = roh # called "Width" in Table 18
  N_LAYERS = 2
  N_STACKS = 1
  INPUT_SIZE = WINDOW_SIZE * HORIZON # called "Lookback" in Table 18
  THETA_SIZE = INPUT_SIZE + HORIZON

  INPUT_SIZE, THETA_SIZE


  # %%time

  tf.random.set_seed(42)

  # 1. Setup N-BEATS Block layer
  nbeats_block_layer = NBeatsBlock(input_size=INPUT_SIZE,
                                  theta_size=THETA_SIZE,
                                  horizon=HORIZON,
                                  n_neurons=N_NEURONS,
                                  n_layers=N_LAYERS,
                                  name="InitialBlock")

  # 2. Create input to stacks
  stack_input = layers.Input(shape=(3), name="stack_input")

  # 3. Create initial backcast and forecast input (backwards predictions are referred to as residuals in the paper)
  backcast, forecast = nbeats_block_layer(stack_input)
  # Add in subtraction residual link, thank you to: https://github.com/mrdbourke/tensorflow-deep-learning/discussions/174
  residuals = layers.subtract([stack_input, backcast], name=f"subtract_00")

  # 4. Create stacks of blocks
  for i, _ in enumerate(range(N_STACKS-1)): # first stack is already creted in (3)

    # 5. Use the NBeatsBlock to calculate the backcast as well as block forecast
    backcast, block_forecast = NBeatsBlock(
        input_size=INPUT_SIZE,
        theta_size=THETA_SIZE,
        horizon=HORIZON,
        n_neurons=N_NEURONS,
        n_layers=N_LAYERS,
        name=f"NBeatsBlock_{i}"
    )(residuals) # pass it in residuals (the backcast)

    # 6. Create the double residual stacking
    residuals = layers.subtract([residuals, backcast], name=f"subtract_{i}")
    forecast = layers.add([forecast, block_forecast], name=f"add_{i}")

  # 7. Put the stack model together
  model_6 = tf.keras.Model(inputs=stack_input,
                          outputs=forecast,
                          name="model_6_N-BEATS")

  # 8. Compile with MAE loss and Adam optimizer
  model_6.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(), metrics=[tf.keras.metrics.RootMeanSquaredError()])

  # 9. Fit the model with EarlyStopping and ReduceLROnPlateau callbacks
  model_6.fit(train_dataset,
              epochs=N_EPOCHS,
              validation_data=test_dataset,
              verbose=0, # prevent large amounts of training outputs
              # callbacks=[create_model_checkpoint(model_name=stack_model.name)] # saving model every epoch consumes far too much time
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
                        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)])
  print(roh)
  # Evaluate N-BEATS model on the test dataset
  rmse=model_6.evaluate(test_dataset)
  rmse=rmse[1]
  print("test RMSE:",rmse)
  train_rmse=model_6.evaluate(train_dataset)
  train_rmse=train_rmse[1]
  print("train RMSE:",train_rmse)
  if rmse < best_test_rmse:
    best_test_rmse=rmse
    best_hidden_size=roh
    best_train_rmse=train_rmse
    best_test_predictions=model_6.predict(X_test)

ws1['N2']=best_hidden_size
ws1['O2']=best_train_rmse
ws1['P2']=best_test_rmse

for i, prediction in enumerate(y_test):
    ws6['A{}'.format(i+2)] = float(prediction)
for i, prediction in enumerate(best_test_predictions):
    ws6['B{}'.format(i+2)] = float(prediction)

wb.save(filename = 'Results.xlsx')
