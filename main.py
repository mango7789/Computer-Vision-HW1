from __init__ import *
from full_connect_network import FullConnectNet
from solve import Solver
from utils import download_minist

# default parameters
default_params = {
    'hidden_dims': [128, 64], 
    'activation': ['relu'],
    'reg': 1e-2,
    'update_rule': 'sgd',
    'optim_config': {
        'learning_rate': 1e-3
    },
    'lr_decay': 0.9
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train and test a three layer neural network on the Fashion-MNIST dataset\n")
    subparsers = parser.add_subparsers(title='mode', dest='mode')
    
    #######################################################################
    #                             train                                   #
    #######################################################################

    train_parser = subparsers.add_parser('train')
    # parameters of the three layer neural network
    train_parser.add_argument("--hidden_dims" , type=int  , nargs='+', default=default_params['hidden_dims'], help="Sizes of the hidden layers, default is [128, 64].")
    train_parser.add_argument("--activation"  , type=str  , nargs='+', default=default_params['activation'] , 
                            help="Activation functions, the length should be 1 or equal to the the hidden dims, can choose from ['relu', 'tanh', 'sigmoid']."
                    )
    train_parser.add_argument("--reg"         , type=float,            default=default_params['reg']        , help="Regularization strength of the l_2 penalty, default is 0.01")
    train_parser.add_argument("--weight_scale", type=float,            default=0.01                         , help="Weight scale of the initial weigh matrix, default is 0.01.")
    
    # parameters of the solver
    train_parser.add_argument("--epochs"       , type=int  , default=10                                               , help="Number of training epochs, default is 10.")
    train_parser.add_argument("--iters"        , type=int  , default=6000                                             , help="Number of training iterations, defalut is 6000.")
    train_parser.add_argument("--update_rule"  , type=str  , default=default_params['update_rule']                    , 
                            help="Update rule in training, default is 'sgd', can choose from ['sgd', 'sgd_momentum', 'adam', 'rmsprop']."
                    )
    train_parser.add_argument("--learning_rate", type=float, default=default_params['optim_config']['learning_rate']  , help="Learning rate of the training, defalut is 1e-3.")
    train_parser.add_argument("--lr_decay"     , type=float, default=default_params['lr_decay']                       , help="Learning rate decay, default is 0.9.")
    train_parser.add_argument("--batch_size"   , type=int  , default=64                                               , help="Batch size, default is 64.")
    train_parser.add_argument("--save"         , type=bool , default=False                                            , help="The trained model should be saved or not, default is False.")
    
    #######################################################################
    #                             test                                    #
    #######################################################################
    
    test_parser = subparsers.add_parser('test')
    test_parser.add_argument("--path", type=str, default='fcnn.npz', help="The path of the trained model in dir ./model.")
    
    args = parser.parse_args()
    
    data = download_minist()
    
    if args.mode.lower() == 'train':
        three_layer_model = FullConnectNet(
            hidden_dims=args.hidden_dims, 
            activation=args.activation, 
            reg=args.reg, 
            weight_scale=args.weight_scale
        )
        
        three_layer_net = Solver(
            model=three_layer_model, 
            data=data,
            batch_size=args.batch_size,
            update_rule=args.update_rule, 
            optim_config={
                'learning_rate': args.learning_rate
            }, 
            lr_decay=args.lr_decay, 
            num_epochs=args.epochs,
        )
        
        if args.save:
            three_layer_net.save("fcnn_args")

        three_layer_net.train() 
        
    elif args.mode.lower() == 'test':
        
        three_layer_trained_model = Solver(FullConnectNet(), {})
        three_layer_trained_model.load(args.path)
        
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        accuracy_table, confusion_matrix = three_layer_trained_model.test_accuracy_table(
                data['X_val'], data['y_val']
            )
        table_names = class_names + ['Total']
        table_headers = ['Accuracy(Recall)', 'Precision', 'F1-score']
        # print the table
        print(tabulate(accuracy_table, headers=table_headers, showindex=table_names, tablefmt='grid'))
    else:
        raise ValueError("Please choose a vaild mode from ['train', 'test']!")