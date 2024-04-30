from __init__ import *
from solve import Solver

def download_minist() -> Dict:
    """
    Download the minist dataset from the website and save them in the local directory `data` .

    Return: 
    - data: a dictionary containing the keys ["X_train", "y_train", "X_val", "y_val"] and corresponding
      images and labels.
    """
    urls = {
        "train_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
        "train_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
        "test_images": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
        "test_labels": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
    }

    download = not os.path.exists('data')

    if download:
        os.makedirs('data')

    train_images = download_and_extract_data(urls["train_images"], os.path.join("data", "train-images-idx3-ubyte.gz"), download)
    train_labels = download_and_extract_data(urls["train_labels"], os.path.join("data", "train-labels-idx1-ubyte.gz"), download)
    test_images = download_and_extract_data(urls["test_images"], os.path.join("data", "t10k-images-idx3-ubyte.gz"), download)
    test_labels = download_and_extract_data(urls["test_labels"], os.path.join("data", "t10k-labels-idx1-ubyte.gz"), download)

    data = {
        'X_train': train_images,
        'y_train': train_labels,
        'X_val': test_images,
        'y_val': test_labels
    }

    return data

def download_and_extract_data(url: str, file_name: str, download=True):
    """
    Download the data from the given url, load it as a numpy array(matrix) and reshape it as size 28*28.

    Input:
    - url: the url of the dataset
    - file_name: the storage path of the dataset
    - download: whether to download the dataset, defaule is True
    """
    if download:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=file_name) as progress_bar:
            def reporthook(block_size, total_size):
                if total_size > 0:
                    progress_bar.total = total_size
                    progress_bar.update(block_size)
            
            urllib.request.urlretrieve(url, file_name, reporthook=reporthook)
    
    with gzip.open(file_name, 'rb') as f:
        if 'images' in file_name:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
            data = data.reshape((-1, 28, 28))
        else:
            data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    return data
    
def plot_stats_single(net: Solver, window_size: int=100):
    """
    Plot the loss function and train / validation accuracies for one single net.
    """
    plt.subplot(2, 1, 1)
    if net.val_loss_hist == []:
        plt.plot(net.train_loss_hist, 'o', label='discrete loss')
        plt.plot([sum(net.train_loss_hist[i:i+window_size])/window_size for i in range(len(net.train_loss_hist)-window_size)], 'red', label='moving average')
    else: 
        plt.plot([sum(net.train_loss_hist[i:i+window_size])/window_size for i in range(len(net.train_loss_hist)-window_size)], 'red', label='train')
        plt.plot([sum(net.val_loss_hist[i:i+window_size])/window_size for i in range(len(net.val_loss_hist)-window_size)], 'purple', label='val')
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(net.train_acc_hist, 'o-', label='train')
    plt.plot(net.val_acc_hist, 'o-', label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()

    plt.gcf().set_size_inches(10.5, 9)
    plt.show()

def plot_acc_multi(nets: List[Solver]):
    """
    Plot the training and validation accuracies for multiple nets in one image.
    """
    plt.subplot(1, 2, 1)
    for net in nets:
        plt.plot(net.train_acc_hist, label=str(net))
    plt.title('Train accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')

    plt.subplot(1, 2, 2)
    for net in nets:
        plt.plot(net.val_acc_hist, label=str(net))
    plt.title('Validation accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()

    plt.gcf().set_size_inches(14, 5)
    plt.show()

def show_net_weights(net: Solver):
    """
    Visualize the weights of the network
    """
    weights = [val for key, val in net.model.params.items() if key[0] == 'W']
    biases = [val for key, val in net.model.params.items() if key[0] == 'b']
    
    num_params = len(weights)
    fig, axs = plt.subplots(num_params, 2, figsize=(15, 15))
    
    # plot weights and biases
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        # plot weight matrix
        img = axs[index, 0].imshow(weight, cmap='viridis', vmin=np.min(weight), vmax=np.max(weight), aspect='auto')
        plt.colorbar(img, ax=axs[index, :].ravel().tolist())
        axs[index, 0].set_title('W{}'.format(index + 1))
        
        # plot bias vector
        axs[index, 1].bar(np.arange(bias.shape[0]), bias)
        axs[index, 1].set_title('b{}'.format(index + 1))
        
    plt.show()

def train_val_split(data: Dict, k: int=5):
    """
    Conduct a k-fold split on the Fashion-MINIST dataset.
    """
    train_samples = len(data['X_train'])
    val_samples = train_samples // k
    # find the indices of the validation samples
    indices = np.arange(train_samples)
    np.random.shuffle(indices)
    fold_indices = [indices[i * val_samples:(i + 1) * val_samples] for i in range(k)]
    # assign the remaining indices to the last fold
    if train_samples % k != 0:
        fold_indices[-1] = np.concatenate((fold_indices[-1], indices[k * val_samples:]))
    return fold_indices

def image_clamp(images_list: List):
    """
    Clamp the gray value of images to get a better visualization of them
    """
    image_shape = images_list[0][0].shape
    reshape_size = int(math.floor(math.sqrt(image_shape[1])))
    return  [
                [
                np.resize(np.interp(image, (np.min(image), np.max(image)), (0, 1)), (reshape_size, reshape_size))
                for image in images
            ]
            for images in images_list
        ]

def plot_images(images_list: List, class_names: List, clamp: bool=True):
    """
    Visualize the Fashion-MNIST dataset in a grid.
    """
    fig, axes = plt.subplots(10, 11, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)  
    for i, images in enumerate(image_clamp(images_list) if clamp else images_list):
        axes[i, 0].text(0.5, 0.5, class_names[i], ha='center')
        axes[i, 0].axis('off')
        for j in range(10):
            ax = axes[i, j + 1]
            ax.imshow(images[j], cmap='gray')
            ax.axis('off')
    plt.show()
