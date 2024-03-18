import torch
from torch import nn
from torch import linalg as LA
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torch.utils import data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def transform_and_choose_train_data_for_scores(size_for_train_set = 500):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.view(-1))
        ])

    generator = torch.Generator().manual_seed(69)

    train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='../data', train=False, transform=transform, download=True)

    random_train = torch.randperm(len(train_dataset), generator=generator)[:size_for_train_set]
    random_test = torch.randperm(len(test_dataset), generator=generator)[:10]

    train_dataloader = DataLoader(train_dataset, batch_size=1)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    train_examples_for_if_scores = data.Subset(train_dataloader.dataset, indices=random_train)
    train_examples_for_if_scores = DataLoader(train_examples_for_if_scores, batch_size=1)
    test_examples_for_if_scores = data.Subset(test_dataloader.dataset, indices=random_test)
    test_examples_for_if_scores = DataLoader(test_examples_for_if_scores, batch_size=1)


    return train_examples_for_if_scores, test_examples_for_if_scores, random_train, random_test


def get_test_if_scores_on_chosen_example():
    return






class PBRFOptimizer(nn.Module):

    def __init__(self,
                base_model_weights,
                base_model,
                current_model,
                damping: 0.001,
                learning_rate: 0.1,
                criterion: None
                ):

        super(PBRFOptimizer, self).__init__()


        self.theta_s = base_model_weights
        self.base_model = base_model
        self.model = current_model
        self.damping = damping
        self.learning_rate = learning_rate
        self.criterion = criterion

    def get_bregman_divergance(self, examples_for_bregman_divergence,
                               original_optimizer):
        bregman_div = 0

        for inputs, labels in examples_for_bregman_divergence:

            original_optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs_curr_wrt_pbrf_optimizer = self.model(inputs)
            outputs_curr_wrt_original_optimizer = self.base_model(inputs)
            outputs_curr_wrt_original_optimizer.retain_grad()

            loss_pbrf_curr = self.get_pred_loss(outputs_curr_wrt_pbrf_optimizer, labels)
            loss_original_curr = self.get_pred_loss(outputs_curr_wrt_original_optimizer, labels)

            loss_original_curr.backward()

            # loss_original_transpose = loss_original.t()
            gradient_loss_original = outputs_curr_wrt_original_optimizer.grad
            final_bregman_term_loss_grads = torch.matmul(gradient_loss_original,
                                            (outputs_curr_wrt_pbrf_optimizer - outputs_curr_wrt_original_optimizer).t())

            final_bregman_term = loss_pbrf_curr.item() - loss_original_curr.item() - final_bregman_term_loss_grads.item()

            outputs_curr_wrt_original_optimizer.grad = None

            bregman_div+=final_bregman_term

        return bregman_div/20

    def get_pred_loss(self, model_output, labels):

        try:
            loss = self.criterion(model_output, labels)
        except:
            loss = self.criterion(model_output, torch.nn.functional.one_hot(labels, num_classes=10).float())

        return loss

    def step(self, outputs_wrt_pbrf_optimizer = '',
             outputs_wrt_original_optimizer = '',
             labels = '',
             examples_for_bregman_divergence = '',
             original_optimizer= ''):

        # outputs_wrt_original_optimizer.retain_grad()
        loss_pbrf = self.get_pred_loss(outputs_wrt_pbrf_optimizer,  labels)
        # loss_original = self.get_pred_loss(outputs_wrt_original_optimizer,  labels)
        #
        # loss_original.backward()

        # gradient_loss_original = outputs_wrt_original_optimizer.grad
        # final_bregman_term_loss_grads = torch.matmul(gradient_loss_original,
        #                                   (outputs_wrt_pbrf_optimizer - outputs_wrt_original_optimizer).t())
        #
        #
        # final_bregman_term = loss_pbrf.item() - loss_original.item() - final_bregman_term_loss_grads.item()

        outputs_wrt_original_optimizer.grad = None


        difference_in_params = (self.damping/2)*(LA.matrix_norm(self.model.fc1.weight - self.theta_s))


        final_bregman_term = self.get_bregman_divergance(examples_for_bregman_divergence, original_optimizer)

        loss_alternate = final_bregman_term  - loss_pbrf/500 + difference_in_params
        return loss_alternate

def test(model, test_loader, criterion, get_preds_only=False, train_loader=None):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not get_preds_only:
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                # Compute cross-entropy loss
                try:
                    loss = criterion(outputs, labels)
                except:
                    # Compute mean squared error
                    loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())

                # Combine the two losses
                # loss = loss_ce #+ loss_mse

                total_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total

        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")
        return avg_loss, accuracy
    else:
        all_preds_array = []
        output_grads = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            all_preds_array.append(outputs)
            outputs.retain_grad()

            # Compute cross-entropy loss
            try:
                loss = criterion(outputs, labels)
            except:
                # Compute mean squared error
                loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())

            loss.backward()

            output_grads.append(outputs.grad)

        return all_preds_array, output_grads



def validate(model, val_loader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            # Compute cross-entropy loss
            try:
                loss = criterion(outputs, labels)
            except:
                # Compute mean squared error
                loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())

            total_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy
