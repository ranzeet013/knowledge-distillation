import torch
import torch.nn.functional as F
from config import TEMPERATURE, ALPHA

def distillation_loss(y, labels, teacher_logits, temp=TEMPERATURE, alpha=ALPHA):
    """
    Computes the knowledge distillation loss.
    
    Combines:
        - Hard target loss (cross-entropy with true labels)
        - Soft target loss (KL divergence with teacher predictions)
    
    Args:
        y (Tensor): Student model logits.
        labels (Tensor): Ground truth labels.
        teacher_logits (Tensor): Teacher model logits.
        temp (float): Temperature for softening the logits.
        alpha (float): Weight between hard and soft loss.
    
    Returns:
        Tensor: Weighted sum of hard and soft losses.
    """
    hard_loss = F.cross_entropy(y, labels)
    soft_loss = F.kl_div(
        F.log_softmax(y / temp, dim=1),
        F.softmax(teacher_logits / temp, dim=1),
        reduction='batchmean'
    ) * (temp ** 2)

    return alpha * hard_loss + (1 - alpha) * soft_loss


def get_teacher_logits(model, loader, device):
    """
    Extracts logits from the teacher model over a given dataset.

    Args:
        model (nn.Module): Trained teacher model.
        loader (DataLoader): DataLoader for dataset.
        device (torch.device): Computation device.

    Returns:
        Tensor: Concatenated logits from teacher.
    """
    model.eval()
    logits = []

    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            output = model(data)
            logits.append(output)

    return torch.cat(logits)


class DistillDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for distillation training.

    Returns:
        data, target, teacher_logits for each sample.
    """
    def __init__(self, dataset, teacher_logits):
        self.dataset = dataset
        self.teacher_logits = teacher_logits

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, self.teacher_logits[index]

    def __len__(self):
        return len(self.dataset)
