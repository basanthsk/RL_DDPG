import torch.nn as nn
import torch.optim as optim
import torch

class StateRepresentation(nn.Module):
    def __init__(self, users_num, items_num,state_size,embedding_dim=100):
        super(StateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(users_num, embedding_dim)
        self.game_embedding = nn.Embedding(items_num, embedding_dim)
        self.wav = nn.Conv1d(state_size, 1, 1)
        self.optimizer = optim.Adam(self.parameters(),lr=1e-5, weight_decay = 1e-3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.initialize()
       

    def forward(self, user,games):
        user_embedding = self.user_embedding(user).unsqueeze(0)
        game_embedding = self.game_embedding(games).unsqueeze(0)
        game_embedding = game_embedding / self.embedding_dim
        wav = self.wav(game_embedding)
        wav = wav.squeeze(1)
        user_wav = torch.mul(user_embedding, wav)
        concat = torch.cat((user_embedding, user_wav, wav), dim=1)
        return concat.view(concat.size(0), -1)

    def initialize(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.game_embedding.weight, std=0.01)
        nn.init.xavier_uniform_(self.wav.weight)
        nn.init.constant_(self.wav.bias, 0)
