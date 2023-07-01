import numpy as np
import torch
from src.state_representation import StateRepresentation


class GameRecommendationEnv:
    def __init__(self, users_dict,user_nums,game_nums,state_size = 5,  max_steps=10):
        self.embedding_dims = 100
        self.users_dict = users_dict
        self.user_nums = user_nums
        self.game_nums = game_nums
        self.max_steps = max_steps
        self.state_size = state_size
        self.available_users = self._get_available_users()
        self.q  = StateRepresentation(user_nums, game_nums,state_size)
        self.done_count = 80
        self.reset()
        
    def reset(self):
        self.user = np.random.choice(self.available_users)
        self.user_game_rating = {data[0]:data[1] for data in self.users_dict[self.user]}
        self.items = [data[0] for data in self.users_dict[self.user][:self.state_size]]
        self.done = False
        
        self.recommended_games = set(self.items)
        state = self.q(torch.tensor(self.user),torch.tensor(self.items))
        return state, self.done
    
    def _get_available_users(self):
        available_users = []
        for userid in self.users_dict:
            if len(self.users_dict[userid]) > self.state_size:
                available_users.append(userid)
        return available_users
        
    def step(self, action):
        reward = 0
        actions = self.recommend_new_game(action)
        rewards =[]
        new_recommendatins =[]
        # print(self.user_game_rating.keys())
        
        for act in actions:
            if act in self.user_game_rating.keys():
                new_recommendatins.append(act)
                if self.user_game_rating[act]>3:
                    rewards.append((self.user_game_rating[act] - 3)/2) #positive reward
            else:
                rewards.append(-0.5)   # Nagative reward
            self.recommended_games.add(act)
        if len(rewards) > 0 and max(rewards) >0:
            self.items = self.items[len(new_recommendatins):] + new_recommendatins
        reward = np.sum(rewards)
        if len(self.recommended_games) > self.done_count or len(self.recommended_games) > len(self.user_game_rating.keys()):
            self.done = True 
        state = self.q(torch.tensor(self.user),torch.tensor(self.items))
        return state, reward, self.done, self.recommended_games
        
    def get_items_names(self, items_ids):
        items_names = []
        for id in items_ids:
            try:
                items_names.append(self.items_id_to_name[str(id)])
            except:
                items_names.append(list(['Not in list']))
        return items_name
    
  
    def recommend_new_game(self, action):
        
        rec_games = torch.tensor(list(set(i for i in range(self.game_nums)) - self.recommended_games))
        games_ebs = self.q.game_embedding(rec_games)
        action = torch.tensor(action.reshape(-1,1))
        item_indices = torch.argsort(torch.mm(games_ebs, action).permute(1,0))[0][-2:]
        #In case need of argmax use below
        # item_idx = torch.argmax(torch.mm(games_ebs, action), dim=0)
        return rec_games[item_indices].cpu().detach().numpy()