import torch
import torch.nn.functional as F

class InfoNCE_with_filtering:
    def __init__(self, temperature=0.7, threshold_selfsim=0.8):
        self.temperature = temperature
        self.threshold_selfsim = threshold_selfsim

    def __call__(self, x, y, sent_emb=None):
        bs, device = x.size(0), x.device
        # x: [bs, latent_dim]
        # y: [bs, 2, latent_dim]

        # Normalize embeddings
        x_norm = F.normalize(x, dim=-1).unsqueeze(1)  # [bs, 1, latent_dim]
        y_norm = F.normalize(y, dim=-1)               # [bs, 2, latent_dim]

        # Compute similarities
        sim_matrix = torch.bmm(x_norm, y_norm.transpose(1, 2)).squeeze(1)  # [bs, 2]
        sim_matrix /= self.temperature

        # Self-similarity filtering
        if sent_emb is not None and self.threshold_selfsim:
            sent_emb_flat = sent_emb.view(-1, sent_emb.shape[-1])  # [bs*2, latent_dim]
            selfsim = torch.mm(sent_emb_flat, sent_emb_flat.T)     # [bs*2, bs*2]
            selfsim = selfsim - torch.diag(torch.diag(selfsim))
            real_threshold_selfsim = 2 * self.threshold_selfsim - 1
            idx = torch.where(selfsim > real_threshold_selfsim)
            sample_indices = (idx[1] // 2).to(torch.long)
            text_indices = (idx[1] % 2).to(torch.long)
            sim_matrix[sample_indices, text_indices] = -float('inf')

        # Labels
        labels = torch.zeros(bs, dtype=torch.long, device=device)  # [bs]

        # Compute the InfoNCE loss
        # Loss is always > 0, if positive text is closer than negative text, the loss is smaller. Otherwise, the loss is larger.
        total_loss = F.cross_entropy(sim_matrix, labels)

        return total_loss