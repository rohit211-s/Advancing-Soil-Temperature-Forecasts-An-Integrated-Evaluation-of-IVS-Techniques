reducer=umap.UMAP(n_components=3,min_dist=0.3)
embedding=reducer.fit_transform(X_scaled)
split_size = int(len(X) * 0.8)
X_train, y_train = embedding[:split_size], y_scaled[:split_size]
X_test, y_test = embedding[split_size:], y_scaled[split_size:]
len(X_train), len(y_train), len(X_test), len(y_test)
