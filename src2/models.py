import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        v_len, k_len, q_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, v_len, self.heads, self.head_dim)
        keys = keys.reshape(N, k_len, self.heads, self.head_dim)
        queries = query.reshape(N, q_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, q_len, self.heads * self.head_dim
        )
        return self.fc_out(out)

class IDSClassifier(nn.Module):
    """
    Hybrid Deep Learning Classifier: 1D-CNN + Bi-LSTM + Attention.
    Optimized for high-speed network traffic classification.
    """
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=2):
        super(IDSClassifier, self).__init__()
        
        # 1. Feature Interaction Extraction (CNN)
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        # 2. Temporal/Contextual dependency (Bi-LSTM)
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3
        )
        
        # 3. Global Relevance Weighting (Multi-Head Attention)
        self.attention = MultiHeadSelfAttention(embed_size=hidden_dim*2, heads=8)
        
        # 4. Dense Decision Layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x: [Batch, Features]
        if len(x.shape) == 2:
            x = x.unsqueeze(1) # [Batch, 1, Features]
        
        # CNN Phase
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x))) # [Batch, 128, Features]
        
        # Permute for LSTM: [Batch, Seq(Features), Hidden]
        x = x.permute(0, 2, 1)
        
        # LSTM Phase
        lstm_out, _ = self.lstm(x) # [Batch, Features, hidden_dim*2]
        
        # Attention Phase
        att_out = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Aggregation Phase (Global Average Pooling over sequence)
        pooled = torch.mean(att_out, dim=1)
        
        # Classification
        return self.fc_layers(pooled)

class AnomalyVAE(nn.Module):
    """
    Variational Autoencoder for Zero-Day Recognition.
    Learns the 'manifold' of normal operating conditions.
    """
    def __init__(self, input_dim, latent_dim=16):
        super(AnomalyVAE, self).__init__()
        
        # Encoder: Compressing network signature
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder: Reconstructing network signature
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, input_dim),
            nn.Sigmoid() # Assumes RobustScaler + MinMax for VAE normalization
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    # BCE or MSE for reconstruction
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # KL Divergence: Regularizing the latent space
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss
