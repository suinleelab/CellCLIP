"""
Loss calculation functions

[1] Molphenix: https://openreview.net/pdf?id=elA8hwvYAm
[2] CWCL:  https://arxiv.org/pdf/2309.14580
"""

import torch
import torch.nn.functional as F


def s2l_loss(image_features, mol_features, inv_tau, bias):
    """S2L loss for molphenix training."""

    def s2l_dist(zxi):
        """
        Customized similarity function for all pairs (i, j) from [1]:
        arctan(||zxi_i - zxj_j||_2^2 / c) * (4 / pi) - 1
        """
        c = 0.3
        threshold = 0.75

        pi = torch.tensor(torch.pi)

        # Compute pairwise L2 distance using:
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2(x·y)
        squared_norms = (zxi ** 2).sum(dim=1, keepdim=True)
        dist_matrix = squared_norms + squared_norms.T - 2 * (zxi @ zxi.T)
        sim_matrix = 1 - torch.atan(dist_matrix / c) * (4 / pi)

        # Clip similarity to 0 if below threshold.
        sim_matrix = torch.where(
            sim_matrix < threshold, torch.tensor(0.0, device=sim_matrix.device), sim_matrix
        )

        return sim_matrix

    gamma = 1.7
    delta = 0.75

    logits_per_image = inv_tau.exp() * image_features @ mol_features.t()
    logits = logits_per_image.t() + bias

    sim_matrix = s2l_dist(image_features)
    pos = F.logsigmoid(logits)
    neg = F.logsigmoid(-logits)
    n = len(logits)
    loss = sim_matrix * pos + (gamma - delta * sim_matrix) * neg
    loss = -torch.sum(loss) / n

    return loss


def cwcl_loss(images, image_features, text_features, inv_tau, loss_fct_txt):
    """CWCL loss for from [2]"""

    logits_per_image = inv_tau.exp() * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    n = len(logits_per_image)

    ground_truth = torch.arange(n, dtype=torch.long, device=images.device)
    loss_cl = loss_fct_txt(logits_per_text, ground_truth)

    B, C, _ = images.shape

    images_norm = images / (images.norm(dim=-1, keepdim=True) + 1e-8)

    sim_matrix = torch.zeros(B, B, device=images.device)
    # Compute cosine similarity per channel and then average across channels
    for c in range(C):
        channel_features = images_norm[:, c, :]  # (B, D)
        channel_sim = torch.mm(channel_features, channel_features.T)  # (B, B)
        sim_matrix += channel_sim

    sim_matrix = sim_matrix / (C * 2) + 0.5
    # sim_matrix = sim_matrix / 2 + 0.5
    sim_matrix = sim_matrix / (sim_matrix.sum(dim=1, keepdim=True) + 1e-8)

    loss_cwcl = -torch.sum(sim_matrix * torch.log_softmax(logits_per_image, dim=1)) / n

    return loss_cwcl + loss_cl


def bi_cwcl_loss(images, image_features, text_features, inv_tau):
    """CWCL loss for with channel similarity and text embedding similarity"""

    logits_per_image = inv_tau.exp() * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    n = len(logits_per_image)

    text_norm = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
    sim_matrix = text_norm @ text_norm.T
    sim_matrix = sim_matrix / 2 + 0.5

    loss_cl = -torch.sum(sim_matrix * torch.log_softmax(logits_per_text, dim=1)) / n

    B, C, _ = images.shape

    images_norm = images / (images.norm(dim=-1, keepdim=True) + 1e-8)

    sim_matrix = torch.zeros(B, B, device=images.device)
    # Compute cosine similarity per channel and then average across channels
    for c in range(C):
        channel_features = images_norm[:, c, :]  # (B, D)
        channel_sim = torch.mm(channel_features, channel_features.T)  # (B, B)
        sim_matrix += channel_sim

    sim_matrix = sim_matrix / (C * 2) + 0.5
    # sim_matrix = sim_matrix / 2 + 0.5
    sim_matrix = sim_matrix / (sim_matrix.sum(dim=1, keepdim=True) + 1e-8)

    loss_cwcl = -torch.sum(sim_matrix * torch.log_softmax(logits_per_image, dim=1)) / n

    return loss_cwcl + loss_cl


def compute_mahalanobis_similarity(images, reg=1e-4):
    """Compute the Mahalanobis similarity matrix for each channel."""

    B, C, D = images.shape
    dist_matrix = torch.zeros(B, B, device=images.device)

    for c in range(C):
        channel_features = images[:, c, :]  # (B, D)

        # Center features
        mean = channel_features.mean(dim=0, keepdim=True)
        X_centered = channel_features - mean  # (B, D)

        # Diagonal covariance approximation: just the variance per feature
        var = X_centered.var(dim=0, unbiased=True) + reg  # (D,)
        inv_var = 1.0 / var  # (D,)

        # Compute Mahalanobis distance (diagonal case)
        x_weighted = X_centered * inv_var  # (B, D)
        diag_term = (X_centered * x_weighted).sum(dim=1)  # (B,)

        # Compute pairwise distance matrix (no unsqueeze needed)
        cross_term = X_centered @ x_weighted.T  # (B, B)
        mahal_dist = diag_term[:, None] - 2 * cross_term + diag_term[None, :]  # (B, B)

        mahal_dist.fill_diagonal_(0.0)  # ensure zero on the diagonal

        dist_matrix += mahal_dist

    alpha = -1.0 / D

    # Exponential decay to convert distance to similarity, normalize across channels
    sim_matrix = F.sigmoid(alpha * dist_matrix / C)
    # sim_matrix = torch.where(
    #     sim_matrix < 0.5, torch.tensor(0.0, device=sim_matrix.device), sim_matrix
    # )
    sim_matrix = sim_matrix / torch.sum(sim_matrix, dim=1, keepdim=True)

    return sim_matrix


def cwcl_ma_loss(images, image_features, text_features, inv_tau, loss_fct_txt):
    """CWCL loss with Mahalanobis-based similarity instead of cosine similarity."""

    # Compute contrastive loss (same as original)
    logits_per_image = inv_tau.exp() * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()
    n = len(logits_per_image)
    ground_truth = torch.arange(n, dtype=torch.long, device=images.device)
    loss_cl = loss_fct_txt(logits_per_text, ground_truth)

    # Compute Mahalanobis similarity
    # sim_matrix = compute_mahalanobis_similarity(images)

    text_norm = F.normalize(text_features, dim=1)
    sim_matrix = text_norm @ text_norm.T

    sim_matrix = sim_matrix / 2 + 0.5
    sim_matrix = sim_matrix / (sim_matrix.sum(dim=1, keepdim=True) + 1e-8)

    # # Compute Mahalanobis-weighted CWCL loss
    loss_cwcl = -torch.sum(sim_matrix * torch.log_softmax(logits_per_image, dim=1)) / n

    return loss_cwcl + loss_cl


def infoLOOB_loss(x, y, i, inv_tau):
    """Clip loss calculation"""

    tau = 1 / inv_tau
    k = x @ y.T / tau

    positives = -torch.mean(torch.sum(k * i, dim=1))

    # For logsumexp the zero entries must be equal to a very large negative number
    large_neg = -1000.0
    arg_lse = k * torch.logical_not(i) + i * large_neg
    negatives = torch.mean(torch.logsumexp(arg_lse, dim=1))

    return tau * (positives + negatives)


def cloob(image_features, text_features, inv_tau, hopfield_layer):
    """Cloob loss calculation"""

    p_xx, p_yy, p_xy, p_yx = hopfield_retrieval(
        image_features, text_features, hopfield_layer
    )
    identity = torch.eye(p_xx.shape[0]) > 0.5
    i = identity.to(p_xx.device)

    loss_img = infoLOOB_loss(p_xx, p_xy, i, inv_tau=inv_tau)
    loss_txt = infoLOOB_loss(p_yy, p_yx, i, inv_tau=inv_tau)

    return loss_img + loss_txt


def clip(image_features, text_features, inv_tau, loss_fct_img, loss_fct_txt):
    """Clip loss calculation"""
    logits_per_image = inv_tau.exp() * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    ground_truth = torch.arange(len(logits_per_image)).long().to(logits_per_image.device)

    loss_img = loss_fct_img(logits_per_image, ground_truth) / 2
    loss_txt = loss_fct_txt(logits_per_text, ground_truth) / 2

    return loss_img + loss_txt


def sigmoid_loss(image_features, text_features, inv_tau, bias):
    """Sigmoid loss for CLIP pretraining"""
    logits_per_image = inv_tau.exp() * image_features @ text_features.t()
    logits = logits_per_image.t() + bias

    n = len(logits)
    # -1s with diagonal 1s
    labels = (2 * torch.eye(n) - torch.ones(n)).long().to(image_features.device)
    loss = -F.logsigmoid(labels * logits).mean()

    return loss


def hopfield_retrieval(image_features, text_features, hopfield_layer):
    """Hopfield retrieval"""
    patterns_xx = hopfield(
        state_patterns=image_features,
        stored_patterns=image_features,
        hopfield_layer=hopfield_layer,
    )
    patterns_yy = hopfield(
        state_patterns=text_features,
        stored_patterns=text_features,
        hopfield_layer=hopfield_layer,
    )
    patterns_xy = hopfield(
        state_patterns=text_features,
        stored_patterns=image_features,
        hopfield_layer=hopfield_layer,
    )
    patterns_yx = hopfield(
        state_patterns=image_features,
        stored_patterns=text_features,
        hopfield_layer=hopfield_layer,
    )

    return patterns_xx, patterns_yy, patterns_xy, patterns_yx


def hopfield(state_patterns, stored_patterns, hopfield_layer):
    """Retrieval function for hopfield network."""
    retrieved_patterns = hopfield_layer.forward(
        (
            stored_patterns.unsqueeze(0),
            state_patterns.unsqueeze(0),
            stored_patterns.unsqueeze(0),
        )
    ).squeeze()
    # Row vectors -> dim=1 to normalize the row vectors
    retrieved_patterns = retrieved_patterns / retrieved_patterns.norm(dim=1, keepdim=True)
    return retrieved_patterns
