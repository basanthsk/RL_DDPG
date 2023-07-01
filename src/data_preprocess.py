import pandas as pd


class Preprocessing:
    def __init__(self,data_path):
        self.df = pd.read_csv(data_path)
        self.prepare_data()
        
    def prepare_data(self):
        self.df['game_id'] = pd.factorize(self.df['item_id'])[0] + 1
        self.data = self.df[['user_id','game_id','rating','training_date']]
        #mapping game names and ids into dictionary
        self.game_id_name_dict = dict(zip(self.df['game_id'], self.df['item_id']))
        self.user_game_history()
        self.users_num = max(set(self.data["user_id"]))+1
        self.items_num = max(set(self.data["game_id"]))+1
        
        
    def split_train_test(self,ratio = 0.8):
        train_data = data.sample(frac=0.8, random_state=16)
        test_data = data.drop(train_data.index).values.tolist()
        train_data = train_data.values.tolist()  
        train_matrix = self._create_user_item_matrix(train_data)
        test_matrix = self._create_user_item_matrix(test_data)
        
        return train_matrix, test_mat 
    
    def user_game_history(self):
        ratings_df = self.data.sort_values(by='training_date', ascending=True)
        self.users_dict= dict()
        self.users_item_history = dict()
        for user_id in set(ratings_df['user_id']):
            self.users_item_history[user_id] = []
            self.users_dict[user_id] =[]
        for _, row in ratings_df.iterrows():
            self.users_dict[row['user_id']].append((row['game_id'], row['rating']))
            if row['rating'] >3:
                self.users_item_history[row['user_id']].append((row['game_id'], row['rating']))
        self.users_item_history_len = [len(self.users_item_history[user_id]) for user_id in self.users_item_history]
        
  
        
if __name__ == "__main__":
    data_path = "F:/Omdena/cognifit/dataset/AI Models/rating_filtered_ss.csv"
    
    p = Preprocessing(data_path)
    
    

