def plot_comparison(x, y, x_label, y_label, title, colors):
    plt.clf() 

    plt.bar(x, y, color = colors)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()

x = ['PCA_10', 'Original']

plot_comparison(x, [pca_10_score, orig_score], "Type Of Dataset", "Accuracy Score", "Dataset Type vs Accuracy", ["blue", "green"])
plot_comparison(x, [pca_10_test_total_time, orig_test_total_time], "Type Of Dataset", "Time for Inference", "Dataset Type vs Testing Time", ["blue", "green"])
plot_comparison(x, [pca_10_train_total_time, orig_train_total_time], "Type Of Dataset", "Time to Train", "Dataset Type vs Training Time", ["blue", "green"])
