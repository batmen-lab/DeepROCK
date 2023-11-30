import torch, itertools, math, copy
from tqdm import tqdm
import numpy as np
from torch import nn
if torch.__version__.startswith("2"):
    from torch import vmap
    from torch.func import hessian
else:
    from functorch import vmap, hessian
from torch.autograd import grad


def get_torch_loaders(X, Y, batch_size, train_ratio=0.5):
    n = X.shape[0]
    n_train = math.ceil(n*train_ratio)

    X_train = X[:n_train, :]
    Y_train = Y[:n_train, :]
    X_val = X[n_train:, :]
    Y_val = Y[n_train:, :]

    loader_train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.Tensor(X_train), 
            torch.Tensor(Y_train)), batch_size, shuffle=True)
    loader_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.Tensor(X_val), 
            torch.Tensor(Y_val)), batch_size, shuffle=False)

    return {"train": loader_train, "test": loader_val}, \
        {"train": (X_train, Y_train), "test": (X_val, Y_val)}

def evaluate(net, data_loader, criterion, device):
    losses = []
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        loss = criterion(net(inputs), labels).cpu().data
        losses.append(loss)
    return torch.stack(losses).mean()

def train(
    net, data_loaders, optimizer, criterion=nn.MSELoss(reduction="mean"),
    nepochs=100, verbose=False, early_stopping=True, patience=5, decay_const=1e-4,
    device=torch.device("cpu")
):
    if "val" not in data_loaders: early_stopping = False
    patience_counter = 0
    if verbose:
        print("starting to train")
        if early_stopping: print("early stopping enabled")
    def get_reg_loss():
        reg = 0
        for name, param in net.named_parameters():
            if "mlp" in name and "weight" in name:
                reg += torch.sum(torch.abs(param))
        return reg * decay_const

    best_net = None
    best_loss = float("inf")
    train_losses, val_losses, reg_losses = [], [], []
    for epoch in range(nepochs):
        running_loss = 0.0
        run_count = 0
        for i, data in enumerate(data_loaders["train"], 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels).mean()
            reg_loss = get_reg_loss()
            (loss + reg_loss).backward()
            optimizer.step()
            running_loss += loss.item()
            run_count += 1
        
        key = "val" if "val" in data_loaders else "train"
        train_loss = running_loss / run_count
        val_loss = evaluate(net, data_loaders[key], criterion, device)
        l1_reg, l2_reg = 0, 0
        reg_loss = get_reg_loss()
        train_losses.append(train_loss)
        val_losses.append(val_loss.item())
        reg_losses.append(reg_loss.item())
        if verbose:
            print("[epoch %d, total %d] train loss: %.5f, val loss: %.5f"
                % (epoch + 1, nepochs, train_loss, val_loss))
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_net = copy.deepcopy(net)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience and early_stopping:
                if verbose:
                    print("early stopping!")
                break

    return best_net, train_losses, val_losses

class DeepROCK(torch.nn.Module):
        def __init__(self, p, hidden_dims=[64]):
                super(DeepROCK, self).__init__()
                self.p = p
                self.hidden_dims = [self.p] + hidden_dims + [1]
                self.activation = torch.nn.ELU()
                self.Z_weight = nn.Parameter(torch.ones(2 * p))
                # Create MLP layer layers
                mlp_layers = []
                for i in range(len(self.hidden_dims) - 1):
                    mlp_layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
                    if i+1 == len(self.hidden_dims) - 1: continue
                    mlp_layers.append(torch.nn.ELU())
                self.mlp = nn.Sequential(*mlp_layers)

        def forward(self, X):
            X_pink =self.Z_weight.unsqueeze(dim=0) * X
            X_mlp = X_pink[:, :self.p] + X_pink[:, self.p:]
            return self.mlp(X_mlp)

        def _get_W(self):
            with torch.no_grad():
                # Calculate weights from MLP
                layers = list(self.mlp.named_children())
                W = None
                for layer in layers:
                    if isinstance(layer[1], torch.nn.Linear):
                        weight = layer[1].weight.cpu().detach().numpy().T
                        W = weight if W is None else np.dot(W, weight)
                W = W.squeeze(-1)
                return W

        def global_feature_importances(self):
            with torch.no_grad():
                # Calculate weights from MLP
                W = self._get_W()
                # Multiply by Z weights
                Z = self.Z_weight.cpu().numpy()
                feature_imp = Z[:self.p] * W
                knockoff_imp = Z[self.p:] * W
                return np.concatenate([feature_imp, knockoff_imp])

        def get_weights(self):
            weights = []
            for name, param in self.named_parameters():
                if "mlp" in name and "weight" in name:
                    weights.append(param.cpu().detach().numpy())
            return weights

        def global_feature_interactions(self, eps=1e-8, calibrate=True):
            with torch.no_grad():
                weights = self.get_weights()
                w_input = weights[0]
                w_later = weights[-1]
                for i in range(len(weights)-2, 0, -1):
                    w_later = np.matmul(w_later, weights[i])
                Z = self.Z_weight.cpu().numpy()
                interaction_ranking = []
                def inter_func(i, j):
                    w_input_i = Z[i]*w_input[:, i%w_input.shape[1]]
                    w_input_j = Z[j]*w_input[:, j%w_input.shape[1]]
                    strength = np.abs((np.multiply(w_input_i, w_input_j)*w_later).sum())
                    if calibrate:
                        import_i = np.abs((w_input_i*w_later).sum())
                        import_j = np.abs((w_input_j*w_later).sum())
                        strength /= (np.sqrt(import_i*import_j)+eps)
                    interaction_ranking.append(((i, j), strength))

                # original-original (i!=j)
                for comb in itertools.combinations(np.arange(self.p), 2): inter_func(*comb)
                # knockoff-knockoff (i!=j)
                for comb in itertools.combinations(np.arange(self.p, 2*self.p), 2): inter_func(*comb)
                # original-knockoff combinations
                for i in np.arange(self.p):
                    for j in np.arange(self.p, 2*self.p):
                        if (i != j-self.p): inter_func(i, j)
                # knockoff-original combinations
                for i in np.arange(self.p, 2*self.p):
                    for j in np.arange(self.p):
                        if (i != j+self.p): inter_func(i, j)
                interaction_ranking.sort(key=lambda x: x[1], reverse=True)
                return interaction_ranking

        def local_feature_importances(self, X, baseline, steps=50, expected=False):
            """expected: True --> expected hessian, False --> integrated hessian"""
            device = next(self.parameters()).device
            if not torch.is_tensor(X): X = torch.Tensor(X).to(device)
            if not torch.is_tensor(baseline): baseline = torch.Tensor(baseline).to(device)
            if len(X.shape) == 1: X = X.unsqueeze(0)
            if expected and len(baseline.shape)==1: baseline = baseline.unsqueeze(0)
            elif not expected and len(baseline.shape)==2: baseline = baseline.mean(0)
            X.requires_grad = True
            if expected:
                alpha_list = np.random.rand(steps+1)
                rand_idx_list = np.random.choice(baseline.shape[0], steps+1)
                scaled_X = torch.cat([baseline[r]+alpha_list[i]*(X.unsqueeze(1)-baseline[r])
                    for i, r in enumerate(rand_idx_list)], dim=1)
                delta_tensor = (X.unsqueeze(1)-baseline[rand_idx_list])
            else:    
                scaled_X = torch.cat([baseline+(float(i)/(steps))*(X.unsqueeze(1)-baseline)
                    for i in range(steps+1)], dim=1)
                delta_tensor = X.unsqueeze(1)-baseline

            grad_tensor = torch.zeros(scaled_X.shape).to(device)
            for i in range(steps+1):
                particular_slice = scaled_X[:,i]
                batch_output = self(particular_slice)
                grad_tensor[:,i,:] = grad(outputs=batch_output, inputs=particular_slice,
                    grad_outputs=torch.ones_like(batch_output).to(device),
                    create_graph=True)[0]
            attributions = (delta_tensor * grad_tensor).mean(axis=1)
            return attributions.mean(axis=0).detach().cpu().numpy()

        def _local_feature_interactions_helper(self, X, baseline, steps=100, expected=False):
            # setup X
            device = next(self.parameters()).device
            if not torch.is_tensor(X): X = torch.Tensor(X).to(device)
            if not torch.is_tensor(baseline): baseline = torch.Tensor(baseline).to(device)
            if len(X.shape) == 1: X = X.unsqueeze(0)
            if expected and len(baseline.shape)==1: baseline = baseline.unsqueeze(0)
            elif not expected and len(baseline.shape)==2: baseline = baseline.mean(0)
            X.requires_grad = True
            # Setup delta tensor
            if expected:
                alpha_list = np.random.rand(steps+1)
                rand_idx_list = np.random.choice(baseline.shape[0], steps+1)
                scaled_X = torch.cat([baseline[r]+alpha_list[i]*(X.unsqueeze(1)-baseline[r])
                    for i, r in enumerate(rand_idx_list)], dim=1)
                grad_delta_tensor = (X.unsqueeze(1)-baseline[rand_idx_list]).unsqueeze(2)
                hess_delta_tensor = torch.stack([torch.bmm(grad_delta_tensor[i].permute(0, 2, 1), 
                    grad_delta_tensor[i]) for i in range(X.shape[0])])
                grad_delta_tensor = grad_delta_tensor.squeeze(2)                
            else:
                scaled_X = torch.cat([baseline+(float(i)/(steps))*(X.unsqueeze(1)-baseline)
                    for i in range(steps+1)], dim=1)
                alpha_list = [float(i)/steps for i in range(steps+1)]
                grad_delta_tensor = X.unsqueeze(1)-baseline
                hess_delta_tensor = torch.bmm(grad_delta_tensor.permute(0, 2, 1), grad_delta_tensor).unsqueeze(1)
            # Compute integrated gradient & hessian
            grad_tensor = torch.zeros(scaled_X.shape).to(device)
            hess_tensor = torch.zeros((*grad_tensor.shape, X.shape[1])).to(device)
            for i in range(steps+1):
                particular_slice = scaled_X[:,i]
                batch_output = self(particular_slice)
                grad_tensor[:,i,:] = grad(outputs=batch_output, inputs=particular_slice,
                    grad_outputs=torch.ones_like(batch_output).to(device),
                    create_graph=True)[0]
                hess_tensor[:, i, :, :] = vmap(hessian(self))(particular_slice).squeeze()*alpha_list[i]

            attributions = torch.abs((grad_delta_tensor*grad_tensor).sum(axis=1))
            interactions = torch.abs((hess_delta_tensor*hess_tensor).sum(axis=1))
            
            return attributions.detach().cpu().numpy(), interactions.detach().cpu().numpy()

        def local_feature_interactions(
            self, X, baseline, steps=100, eps=1e-8, batch_size=250,
            expected=True, calibrate=True):
            n = X.shape[0]
            attributions_list = []
            interactions_list = []
            for i in tqdm(range(math.ceil(n/batch_size))):
                curr_attributions, curr_interactions = self._local_feature_interactions_helper(
                    X[i*batch_size:(i+1)*batch_size], baseline, steps=steps, expected=expected)
                attributions_list.append(curr_attributions)            
                interactions_list.append(curr_interactions)
            import_arr = np.concatenate(attributions_list, axis=0)
            inter_arr = np.concatenate(interactions_list, axis=0)
            if calibrate:
                denominator_arr = np.sqrt(import_arr[..., None] * import_arr[:, None, :])
                inter_arr /= (denominator_arr+eps)
            inter_arr = inter_arr.mean(0)

            interaction_ranking = []
            def inter_func(i, j):
                score = np.abs(inter_arr[i, j])
                interaction_ranking.append(((i, j), score))
            # original-original (i!=j)
            for comb in itertools.combinations(np.arange(self.p), 2): inter_func(*comb)
            # knockoff-knockoff (i!=j)
            for comb in itertools.combinations(np.arange(self.p, 2*self.p), 2): inter_func(*comb)
            # original-knockoff combinations
            for i in np.arange(self.p):
                for j in np.arange(self.p, 2*self.p):
                    if (i != j-self.p): inter_func(i, j)
            # knockoff-original combinations
            for i in np.arange(self.p, 2*self.p):
                for j in np.arange(self.p):
                    if (i != j+self.p): inter_func(i, j)
            interaction_ranking.sort(key=lambda x: x[1], reverse=True)

            return interaction_ranking

