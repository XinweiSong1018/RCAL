import torch
import torch.nn.functional as F
# Mean Squared Error loss for regression-based emotion prediction
def emo_loss_fn(output, label):
    """
    Compute MSE (Mean Squared Error) loss between predicted and ground truth labels.

    Args:
        output (Tensor): Predicted emotion values (shape: [B, num_emotions])
        label (Tensor): Ground truth emotion values (shape: [B, num_emotions])

    Returns:
        Tensor: Scalar loss value
    """
    loss_fn = torch.nn.MSELoss()
    return loss_fn(output, label)
    

def SupervisedContrastiveLoss(h_v_cls, labels, sigma=0.1):
    """
    Compute supervised contrastive loss for regression-based labels using a Gaussian-weighted cosine similarity.

    Args:
        h_v_cls (Tensor): Visual features after projection (shape: [B, D])
        labels (Tensor): Continuous-valued sentiment labels (shape: [B])
        sigma (float): Width of the Gaussian kernel used to weigh pairwise similarities

    Returns:
        Tensor: Scalar supervised contrastive loss
    """
    batch_size = h_v_cls.shape[0]

    # Normalize feature embeddings to unit vectors
    h_v_cls = torch.nn.functional.normalize(h_v_cls, dim=1)  # (B, D)

    # Compute cosine similarity matrix between all pairs: (B, B)
    similarity_matrix = torch.cosine_similarity(h_v_cls.unsqueeze(1), h_v_cls.unsqueeze(0), dim=-1)

    # Compute pairwise label differences: (B, B)
    labels = labels.contiguous().view(-1, 1)  # (B, 1)
    label_diff = labels - labels.T  # (B, B)

    # Apply a Gaussian kernel over label similarity
    weight_matrix = torch.exp(- (label_diff ** 2) / (2 * sigma ** 2)).to(h_v_cls.device)  # (B, B)

    # Convert cosine similarity into exponential form
    exp_sim = torch.exp(similarity_matrix)  # (B, B)

    # Remove diagonal elements (self-comparisons) by masking with identity matrix
    identity_mask = torch.eye(batch_size, device=h_v_cls.device)
    weight_matrix = weight_matrix * (1 - identity_mask)
    exp_sim = exp_sim * (1 - identity_mask)

    # Compute weighted numerator and denominator for contrastive loss
    numerator = torch.sum(exp_sim * weight_matrix, dim=1)  # (B,)
    denominator = torch.sum(exp_sim, dim=1) + 1e-8          # (B,)

    # Compute loss as negative log of weighted similarity ratio
    loss = -torch.log(numerator / denominator + 1e-8)       # (B,)
    return loss.mean()
