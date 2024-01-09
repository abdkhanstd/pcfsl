import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from io_utils import parse_args

class FewShotPointCloudLearner:
    def __init__(self, z_query, z_proto, params):
        self.params = params
        self.f1 = SIM(z_query).cuda()
        self.f2 = SIM(z_proto).cuda()

        self.f3 = SARF(embedding_dim=params.embedding_length).cuda()
        self.f4 = SARF(embedding_dim=params.embedding_length).cuda()

        self.W1 = nn.Parameter(torch.Tensor(1)).cuda()  # Initialize with a scalar tensor
        self.B1 = nn.Parameter(torch.Tensor(1)).cuda()  # Initialize with a scalar tensor

        self.W2 = nn.Parameter(torch.Tensor(1)).cuda()  # Initialize with a scalar tensor
        self.B2 = nn.Parameter(torch.Tensor(1)).cuda()  # Initialize with a scalar tensor

        self.W3 = nn.Parameter(torch.Tensor(1)).cuda()  # Initialize with a scalar tensor
        self.B3 = nn.Parameter(torch.Tensor(1)).cuda()  # Initialize with a scalar tensor

        self.W4 = nn.Parameter(torch.Tensor(1)).cuda()  # Initialize with a scalar tensor
        self.B4 = nn.Parameter(torch.Tensor(1)).cuda()  # Initialize with a scalar tensor        

        self.reset_parameters()

    def reset_parameters(self):
        # You can use any initialization strategy here
        nn.init.ones_(self.W1)  # Initialize the weight to ones
        nn.init.ones_(self.W2)  # Initialize the weight to ones
        nn.init.ones_(self.W3)  # Initialize the weight to ones
        nn.init.ones_(self.W4)  # Initialize the weight to ones        
        nn.init.zeros_(self.B1)  # Initialize the Baise to zeros
        nn.init.zeros_(self.B2)  # Initialize the weight to zeros
        nn.init.zeros_(self.B3)  # Initialize the Baise to zeros
        nn.init.zeros_(self.B4)  # Initialize the weight to zeros        

    
    def self_interaction_and_attention(self, z_query, z_proto):
        z_query = self.f1(z_query)
        z_proto = self.f2(z_proto)

        #z_proto = (self.f3(z_proto)*self.W1+self.B1) + (z_proto*self.W2+self.B2)
        #z_query = (self.f3(z_query)*self.W3+self.B3) + (z_query*self.W4+self.B4)


        z_query, _ = torch.max(torch.stack([self.f3(z_query), z_query]), dim=0)
        z_proto, _ = torch.max(torch.stack([self.f3(z_proto), z_proto]), dim=0)        


        return z_query, z_proto
    
    
class SARF(nn.Module):
    def __init__(self, embedding_dim=1024, dropout=0.1):
        super(SARF, self).__init__()

        self.embedding_dim = embedding_dim

        # Multi-Head Self-Attention
        self.self_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1, dropout=dropout)

        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim, bias=True),  # Include bias here
        )
        self.init_weights()

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def init_weights(self):
        for layer in self.feedforward:
            if isinstance(layer, nn.Linear):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, support_set):
        # Data Normalization
        support_set = self.normalize_data(support_set)

        # Expand dimensions to match the required input format for nn.Transformer (seq_length = 1)
        support_set = support_set.unsqueeze(1)

        # Multi-Head Self-Attention
        attention_output, _ = self.self_attention(support_set, support_set, support_set)

        # Add Residual and Layer Normalization
        support_set = self.layer_norm1(support_set + self.dropout(attention_output))

        # Feedforward Neural Network
        feedforward_output = self.feedforward(support_set)

        # Add Residual and Layer Normalization
        support_set = self.layer_norm2(support_set + self.dropout(feedforward_output))

        # Remove the extra seq_length dimension and return the attention output
        support_set = support_set.squeeze(1)
        return support_set

    def normalize_data(self, data):
        # Assuming data is a torch tensor, you can use different normalization techniques
        # based on your data characteristics (e.g., min-max scaling, z-score normalization, etc.)
        # Here, we'll perform min-max scaling to bring values to a standard range [0, 1].
        min_val = torch.min(data)
        max_val = torch.max(data)
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data

class SIM(nn.Module):
    def __init__(self, f):
        super(SIM, self).__init__()
        input_dim = f.size(1)
        self.numrows = f.size(0)
        self.linear1 = nn.Linear(input_dim, input_dim, bias=True)
        self.linear2 = nn.Linear(input_dim, input_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.ones_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, f):
        F1 = self.linear1(f)
        F2 = self.linear2(f)
        
        # Apply softmax along the last dimension (columns) to F1 and F2
        F1_softmax = F1.softmax(dim=-1)
        F2_softmax = F2.softmax(dim=-1)

        # Add the original input tensor f with F1_softmax and then with F2_softmax
        output = f + F1_softmax + F2_softmax

        return output
    
import numpy as np
import matplotlib.pyplot as plt

def save_comparison_plot(original_row, sim_output_row):
    # Move the tensors from GPU to CPU and select the first 484 features
    selected_original = original_row.squeeze().cpu()[:484]
    selected_sim_output = sim_output_row.squeeze().cpu()[:484]

    original_matrix = selected_original.view(22, 22).cpu().detach().numpy()
    sim_output_matrix = selected_sim_output.view(22, 22).cpu().detach().numpy()

    # Plot the comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Row")
    plt.imshow(original_matrix, cmap="viridis", aspect="auto")
    plt.grid(False)  # Remove grid lines

    plt.subplot(1, 2, 2)
    plt.title("SIM Output Row")
    plt.imshow(sim_output_matrix, cmap="viridis", aspect="auto")
    plt.grid(False)  # Remove grid lines

    # Save the plot as an image file
    plt.savefig('comparison.png')
    plt.close()  # Close the plot to prevent overlapping

    # Plot the surface view of the matrices
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    x = np.arange(0, 22)
    y = np.arange(0, 22)
    X, Y = np.meshgrid(x, y)

    ax1.plot_surface(X, Y, original_matrix, cmap="viridis")
    ax1.set_title("Original Row Surface View")

    ax2.plot_surface(X, Y, sim_output_matrix, cmap="viridis")
    ax2.set_title("SIM Output Row Surface View")

    plt.tight_layout()

    # Save the surface view plots as image files
    plt.savefig('surface_view_original.png')
    plt.savefig('surface_view_sim_output.png')
    plt.close()  # Close the plot to prevent overlapping
    
class SIM2(nn.Module):
    def __init__(self, f):
        super(SIM2, self).__init__()
        input_dim = f.size(1)
        self.numrows=f.size(0)
        self.linear1 = nn.Linear(input_dim, input_dim, bias=True)
        self.linear2 = nn.Linear(input_dim, input_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.ones_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        
    def forward(self, f):
        outputs = []
        st=0
        for row_idx in range(self.numrows):
            row = f[row_idx, :].unsqueeze(0)
            row_=row
            F1 = self.linear1(row)
            F2 = self.linear2(row)
            row = F1.softmax(dim=-1) + row
            row = F2.softmax(dim=-1) + row 

            outputs.append(row)

        x = torch.cat(outputs, dim=0)

        return x
