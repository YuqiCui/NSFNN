import torch
from sklearn.model_selection import train_test_split
from lib.utils import data_loader


class Train(object):
    def __init__(self, model, criterion, optimizer, early_stop=-1, val_size=-1,
                 epochs=1, random_state=None, batch_size=1, device='cpu', save_path=None, classification=True):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.early_stop = early_stop
        if self.early_stop > 0:
            assert 0 < val_size < 1, "val size must be range of (0, 1) when early stopping is used"
        self.val_size = val_size
        self.epochs = epochs
        self.random_state = random_state
        self.batch_size = batch_size
        self.device = device
        self.model = self.model.to(self.device)
        self.save_path = save_path
        self.classification = classification

    def fit(self, x_train, y_train, x_test, y_test):
        if self.classification:
            return self.fit_classification(x_train, y_train, x_test, y_test)
        else:
            return self.fit_regression(x_train, y_train, x_test, y_test)

    def fit_regression(self, x_train, y_train, x_test, y_test):
        test_loader = data_loader(x_test, y_test, batch_size=1024, shuffle=False, last_long=False)
        if self.early_stop > 0:
            count = 0
            best_valid = float('inf')

        if self.val_size > 0:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size, random_state=self.random_state)
            valid_loader = data_loader(x_val, y_val, batch_size=1024, shuffle=False, last_long=False)
        else:
            valid_loader = None

        train_loader = data_loader(x_train, y_train, batch_size=self.batch_size, shuffle=True, last_long=False)
        for e_ in range(self.epochs):
            tol_loss, tol_batch = 0, 1
            self.model.train()
            for idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if inputs.size(0) == 1:
                    continue  # avoid batch size  = 1

                outs = self.model(inputs)
                loss = self.criterion(outs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tol_loss += loss.item()
                tol_batch = idx
            tol_loss /= max(tol_batch, 1)

            # evaluate MSE for valid_loader and others
            print_info = "[EPOCHS %6d/%d] Train loss: %.4f, " % (e_, self.epochs, tol_loss)
            if valid_loader is not None:
                valid_mse = self.eval_mse(valid_loader)
                print_info += 'Valid MSE: %.4f ' % valid_mse
            test_mse = self.eval_mse(test_loader)
            print_info += 'Test MSE: %.4f ' % test_mse
            print(print_info)
            if self.early_stop > 0:
                if valid_mse < best_valid - 1e-4:
                    best_valid = valid_mse
                    count = 0
                    if self.save_path is not None:
                        torch.save(self.model.state_dict(), self.save_path)
                    results = [valid_mse, test_mse]
                else:
                    count += 1
                    if count > self.early_stop:
                        print('early stopping occurs')
                        self.model.load_state_dict(torch.load(self.save_path))
                        break  # stop training
        return results

    def fit_classification(self, x_train, y_train, x_test, y_test):
        test_loader = data_loader(x_test, y_test, batch_size=1024, shuffle=False)
        if self.early_stop > 0:
            count = 0
            best_valid = 0

        if self.val_size > 0:
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size, random_state=self.random_state)
            valid_loader = data_loader(x_val, y_val, batch_size=1024, shuffle=False)
        else:
            valid_loader = None

        train_loader = data_loader(x_train, y_train, batch_size=self.batch_size, shuffle=True)
        for e_ in range(self.epochs):
            tol_loss, tol_batch = 0, 0
            self.model.train()
            for idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outs = self.model(inputs)
                loss = self.criterion(outs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                tol_loss += loss.item()
                tol_batch = idx
            tol_loss /= tol_batch

            # evaluate ACC and BCA for valid_loader and others
            print_info = "[EPOCHS %6d/%d] Train loss: %.4f, " % (e_, self.epochs, tol_loss)
            if valid_loader is not None:
                valid_acc, valid_bca = self.eval_acc_bca(valid_loader)
                print_info += 'Valid ACC: %.4f, Valid BCA: %.4f, ' % (valid_acc, valid_bca)
            test_acc, test_bca = self.eval_acc_bca(test_loader)
            print_info += 'Test ACC: %.4f, Test BCA: %.4f ' % (test_acc, test_bca)
            print(print_info)
            if self.early_stop > 0:
                if valid_bca > best_valid + 1e-4:
                    best_valid = valid_bca
                    count = 0
                    if self.save_path is not None:
                        torch.save(self.model.state_dict(), self.save_path)
                    results = [valid_acc, valid_bca, test_acc, test_bca]
                else:
                    count += 1
                    if count > self.early_stop:
                        print('early stopping occurs')
                        self.model.load_state_dict(torch.load(self.save_path))
                        break  # stop training
        return results

    def predict_acc_bca(self, x, y):
        loader = data_loader(x, y, batch_size=1024, shuffle=False)
        return self.eval_acc_bca(loader)

    def eval_acc_bca(self, loader):
        # faster when batch size is larger
        self.model.eval()
        n_classes = self.model.n_classes
        class_correct = torch.zeros(n_classes)
        class_count = torch.zeros(n_classes)
        num_data, num_correct = 0, 0
        for s, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            out = self.model(inputs)
            if hasattr(self.model, "firing_branch"):
                out = out[0]
            num_data += inputs.size(0)
            pred_label = torch.argmax(out, dim=1)

            for c in range(n_classes):
                index_correct_c = targets == c
                class_correct[c] += torch.sum(pred_label[index_correct_c] == targets[index_correct_c])
                class_count[c] += torch.sum(index_correct_c)

            num_correct += torch.sum(pred_label == targets).item()
        acc = num_correct / num_data
        bca = torch.mean(class_correct / class_count).cpu().numpy()
        return acc, bca

    def eval_mse(self, loader):
        self.model.eval()
        tol_diff, num_data = 0, 0
        for s, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            out = self.model(inputs)
            tol_diff += torch.pow(out - targets, 2).sum().item()
            num_data += inputs.size(0)
        return tol_diff / num_data
