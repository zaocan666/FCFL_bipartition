import numpy as np 
from utils import read_data
from torch.utils.data import Dataset, DataLoader

class Federated_Dataset(Dataset):
    def __init__(self, X, Y, A):
        self.X = X
        self.Y = Y
        self.A = A

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        A = self.A[index]
        return X, Y, A 

    def __len__(self):
        return self.X.shape[0]

def age_split(age, sensitive_group):
    A = np.zeros_like(age)
    if sensitive_group==4:
        A[np.where(age<=20)]=0
        A[np.where((age>20) & (age<=35))]=1
        A[np.where((age>35) & (age<=50))]=2
        A[np.where((age>50))]=3
    return A

def LoadDataset(args):
    clients_name, groups, train_data, test_data = read_data(args.train_dir, args.test_dir)

    client_train_loads = []
    client_test_loads = []
    args.n_clients = len(clients_name)
    if "adult" in args.dataset:

        def get_sensitive_adult(X):
            if args.sensitive_attr == "race":
                A = X[:,51] # [1: white, 0: other]
                X = np.delete(X, [51, 52, 53, 54, 55], axis = 1)
            elif args.sensitive_attr == "sex":
                A = X[:, 56] # [1: female, 0: male]
                X = np.delete(X, [56, 57], axis = 1)
            elif args.sensitive_attr == "none-race":
                A = X[:, 51]  # [1: white, 0: other]
            elif args.sensitive_attr == "none-sex":
                A = X[:, 56]
            else:
                print("error sensitive attr")
                exit()
            return X, A
        
        def get_sensitive_adult_age(X):
            if args.sensitive_attr == "age":
                age = X[:, 0]
                A = age_split(age, args.sensitive_group)
                X = np.delete(X, [0], axis = 1)
            elif args.sensitive_attr == "race":
                X = np.delete(X, [0], axis = 1)
                A = X[:,51] # [1: white, 0: other]
                X = np.delete(X, [51, 52, 53, 54, 55], axis = 1)
            else:
                print("error sensitive attr")
                exit()
            return X, A

        for client in clients_name:
            X = np.array(train_data[client]["x"]).astype(np.float32)
            Y = np.array(train_data[client]["y"]).astype(np.float32)

            if args.dataset=="adult":
                X, A = get_sensitive_adult(X)
            elif args.dataset=="adult_age":
                X, A = get_sensitive_adult_age(X)
            else:
                raise ValueError()
            
            dataset = Federated_Dataset(X, Y, A)
            client_train_loads.append(DataLoader(dataset, X.shape[0],
            shuffle = args.shuffle,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = args.drop_last))


        for client in clients_name:
            X = np.array(test_data[client]["x"]).astype(np.float32)
            Y = np.array(test_data[client]["y"]).astype(np.float32)

            if args.dataset=="adult":
                X, A = get_sensitive_adult(X)
            elif args.dataset=="adult_age":
                X, A = get_sensitive_adult_age(X)
            else:
                raise ValueError()

            dataset = Federated_Dataset(X, Y, A)
            client_test_loads.append(DataLoader(dataset, X.shape[0],
            shuffle = False,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = args.drop_last)) 

    elif "eicu" in args.dataset:
        def get_sensitive_eicu(client_data):
            if '_age' in args.dataset:
                if args.sensitive_attr == "age":
                    age = client_data['age']
                    A = age_split(age, args.sensitive_group)
                elif args.sensitive_attr == "race":
                    race = client_data['race']
                    A = race[:, 3]
                elif args.sensitive_attr == "race_multi":
                    race = client_data['race']
                    A = np.argmax(race, axis=1)
            return A

        for client in clients_name:
            X = np.array(train_data[client]["x"]).astype(np.float32)
            Y = np.array(train_data[client]["y"]).astype(np.float32)
            A = get_sensitive_eicu(train_data[client])
            
            dataset = Federated_Dataset(X, Y, A)
            client_train_loads.append(DataLoader(dataset, X.shape[0],
            shuffle = args.shuffle,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = args.drop_last))

        for client in clients_name:
            X = np.array(test_data[client]["x"]).astype(np.float32)
            Y = np.array(test_data[client]["y"]).astype(np.float32)
            A = get_sensitive_eicu(train_data[client])

            dataset = Federated_Dataset(X, Y, A)
            client_test_loads.append(DataLoader(dataset, X.shape[0],
            shuffle = False,
            num_workers = args.num_workers,
            pin_memory = True,
            drop_last = args.drop_last)) 

    elif args.dataset == "health":
        for client in clients_name:
            X = np.array(train_data[client]["x"]).astype(np.float32)

            Y = np.array(train_data[client]["y"]).astype(np.float32)

            if args.sensitive_attr == "race":
                A = train_data[client]["race"]
            elif args.sensitive_attr == "sex":
                A = train_data[client]["isfemale"]
            else:
                A = train_data[client]["isfemale"]
            dataset = Federated_Dataset(X, Y, A)
            client_train_loads.append(DataLoader(dataset, X.shape[0],
                                                 shuffle=args.shuffle,
                                                 num_workers=args.num_workers,
                                                 pin_memory=True,
                                                 drop_last=args.drop_last))

        for client in clients_name:
            X = np.array(test_data[client]["x"]).astype(np.float32)
            Y = np.array(test_data[client]["y"]).astype(np.float32)
            if args.sensitive_attr == "race":
                A = test_data[client]["race"]
            elif args.sensitive_attr == "sex":
                A = test_data[client]["isfemale"]
            else:
                A = np.zeros(X.shape[0])

            dataset = Federated_Dataset(X, Y, A)

            client_test_loads.append(DataLoader(dataset, X.shape[0],
                                                shuffle=args.shuffle,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                drop_last=args.drop_last))
    args.n_feats = X.shape[1]
    n_feats = X.shape[1]
    return client_train_loads, client_test_loads, n_feats

