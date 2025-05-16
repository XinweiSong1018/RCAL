import torch
import torch.nn.functional as F

def emo_loss_fn(output, label):
    loss_fn = torch.nn.MSELoss()
    return loss_fn(output, label)
    

def SupervisedContrastiveLoss(h_v_cls, labels, sigma=0.1):
    """
    :param h_v_cls: (B, D) - visual embeddings after projection (final representation)
    :param labels: (B,) - continuous sentiment labels
    :param sigma: float - Gaussian kernel width (smaller = stricter similarity)
    :return: scalar contrastive loss
    """
    batch_size = h_v_cls.shape[0]

    # Normalize embeddings
    h_v_cls = torch.nn.functional.normalize(h_v_cls, dim=1)  # (B, D)

    # Cosine similarity matrix: (B, B)
    similarity_matrix = torch.cosine_similarity(h_v_cls.unsqueeze(1), h_v_cls.unsqueeze(0), dim=-1)

    # Compute label distance matrix
    labels = labels.contiguous().view(-1, 1)  # (B, 1)
    label_diff = labels - labels.T  # (B, B)

    # Gaussian weight based on label similarity
    weight_matrix = torch.exp(- (label_diff ** 2) / (2 * sigma ** 2)).to(h_v_cls.device)  # (B, B)

    # Exponential of cosine similarity
    exp_sim = torch.exp(similarity_matrix)  # (B, B)

    # Avoid self-comparison by masking diagonal to 0
    identity_mask = torch.eye(batch_size, device=h_v_cls.device)
    weight_matrix = weight_matrix * (1 - identity_mask)
    exp_sim = exp_sim * (1 - identity_mask)

    # Compute weighted contrastive loss
    numerator = torch.sum(exp_sim * weight_matrix, dim=1)  # (B,)
    denominator = torch.sum(exp_sim, dim=1) + 1e-8          # (B,)

    loss = -torch.log(numerator / denominator + 1e-8)       # (B,)
    return loss.mean()
