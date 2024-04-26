# Convert your data to tensors
inputs_train = torch.tensor(X_train, dtype=torch.float)
labels_train = torch.tensor(y_train, dtype=torch.float)
inputs_test = torch.tensor(X_test, dtype=torch.float)
labels_test = torch.tensor(y_test, dtype=torch.float)

# Create train and test datasets
train_dataset = TensorDataset(inputs_train, labels_train)
test_dataset = TensorDataset(inputs_test, labels_test)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

best_test_rmse=1
for roh in range(1,21):
  # Create ELM model
  input_dim = inputs_train.shape[1]
  hidden_dim = [roh]
  output_dim = 1  # Set output_size to 1 for a single regression target

  # Create an instance of the MLP model
  model = MLP(input_dim, hidden_dim, output_dim)

  # Define your loss function
  criterion = nn.MSELoss()

  print("hello")
  print(model.parameters())
  # Create an instance of the AdaBelief optimizer
  optimizer = RangerAdaBelief(model.parameters(), lr=1e-2, eps=1e-12, betas=(0.9,0.999),weight_decouple = False)

  # Training loop
  num_epochs = 100
  best_loss = float('inf')  # Initialize with a very high loss
  for epoch in range(num_epochs):
      model.train()
      total_loss = 0
      for batch_idx, (data, target) in enumerate(train_loader):
          data, target = Variable(data), Variable(target)
          optimizer.zero_grad()
          output = model(data)
          loss = criterion(output, target)
          loss.backward()
          optimizer.step()
          total_loss += loss.item()

      avg_loss = total_loss / len(train_loader)
      #print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss}")
      # Check if current loss is the lowest so far
      if avg_loss < best_loss:
          best_loss = avg_loss
          best_model_state = model.state_dict()


  # Load the state of the best model
  model.load_state_dict(best_model_state)

  # Evaluation
  model.eval()

  train_predictions=[]
  with torch.no_grad():
      for batch_idx, (data, target) in enumerate(train_loader):
          data, target = Variable(data), Variable(target)
          output = model(data)
          train_predictions.extend(output.numpy())
  train_predictions = np.array(train_predictions)
  if roh==1:
    labels_train = labels_train.numpy()
  else:
    labels_train = labels_train
  # Load the state of the best model
  train_rmse = np.sqrt(mean_squared_error(labels_train, train_predictions))

  predictions = []
  with torch.no_grad():
      for batch_idx, (data, target) in enumerate(test_loader):
          data, target = Variable(data), Variable(target)
          output = model(data)
          predictions.extend(output.numpy())

  predictions = np.array(predictions)
  if roh==1:
    labels_test = labels_test.numpy()
  else:
    labels_test = labels_test
  rmse = np.sqrt(mean_squared_error(labels_test, predictions))
  print(roh)
  print(f"Test RMSE: {rmse}")
  print("train RMSE:",train_rmse)
  if rmse < best_test_rmse:
    best_test_rmse=rmse
    best_hidden_size=roh
    best_train_rmse=train_rmse
    best_test_predictions=predictions

ws1['H2']=best_hidden_size
ws1['I2']=best_train_rmse
ws1['J2']=best_test_rmse

for i, prediction in enumerate(y_test):
    ws4['A{}'.format(i+2)] = float(prediction)
for i, prediction in enumerate(best_test_predictions):
    ws4['B{}'.format(i+2)] = float(prediction)
wb.save(filename = 'Results.xlsx')
