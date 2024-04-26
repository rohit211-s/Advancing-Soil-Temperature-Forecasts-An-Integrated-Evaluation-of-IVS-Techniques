class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.hidden_layers = nn.ModuleList()
        in_features = input_size
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_features, hidden_size))
            self.hidden_layers.append(nn.ReLU())
            in_features = hidden_size
        self.output_layer = nn.Linear(in_features, output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

best_test_rmse=0.999999
# Step 2: Create and train the MLP model
# Modify the hyperparameters to adjust the model architecture and training process
for roh in range(1,21):
  mlp = MLPRegressor(hidden_layer_sizes=(roh), activation='relu', solver='adam', max_iter=100, random_state=42)
  mlp.fit(X_train, y_train)
  train_predictions = mlp.predict(X_train)
  test_predictions = mlp.predict(X_test)
  train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
  test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
  print(roh)
  print("Training RMSE:", train_rmse)
  print("Testing RMSE:", test_rmse)
  if test_rmse < best_test_rmse:
    best_test_rmse=test_rmse
    best_hidden_size=roh
    best_train_rmse=train_rmse
    best_test_predictions=test_predictions
ws1['B2']=best_hidden_size
ws1['C2']=best_train_rmse
ws1['D2']=best_test_rmse

for i, prediction in enumerate(y_test):
    ws2['A{}'.format(i+2)] = float(prediction)
for i, prediction in enumerate(best_test_predictions):
    ws2['B{}'.format(i+2)] = float(prediction)
wb.save(filename = 'Results.xlsx')
