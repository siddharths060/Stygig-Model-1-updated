import torch
import torch.nn as nn


class OutfitCompatibilityModel(nn.Module):
    """
    Neural network that predicts if two clothing items are compatible
    """

    def __init__(self, embedding_dim=512):
        super(OutfitCompatibilityModel, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, item1_embedding, item2_embedding):

        x = torch.cat([item1_embedding, item2_embedding], dim=1)

        score = self.network(x)

        return score


def predict_compatibility(model, emb1, emb2):
    """
    Predict compatibility score between two clothing items
    """

    with torch.no_grad():

        emb1 = torch.tensor(emb1).float()
        emb2 = torch.tensor(emb2).float()

        score = model(emb1, emb2)

    return score.item()