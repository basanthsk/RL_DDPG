import gym
import numpy  as np
from src.agent import Agent
from src.utils import plot_learning_curve, plot_q_curve
from src.data_preprocess import Preprocessing
from src.env import GameRecommendationEnv

if __name__ == '__main__':
    data_path = "..data/rating_filtered.csv"
    ds = Preprocessing(data_path)
    
    
    env = GameRecommendationEnv(ds.users_dict,ds.users_num,ds.items_num,state_size=5)
    agent = Agent(alpha = 0.0001, beta = 0.001,tau = 0.001,
                  input_dims = [3*env.embedding_dims], batch_size=32, fc1_dims=400,
                  fc2_dims=300,n_actions=env.embedding_dims,state_size=10,
                  users_num = ds.users_num, games_num = ds.items_num)
    n_games = 2000
    
    
    score_history_filename = 'score_history.png'
    score_history_filpath = './plots/'+score_history_filename
    
    qloss_history_filename = 'qloss_history.png'
    q_loss_history_filpath = './plots/'+qloss_history_filename
    
    
    best_score = 0
    score_history = [] 
    q_loss_history = []
    for i in range(n_games):
        state,_ = env.reset()
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(state)
            state_,reward,done,_ = env.step(action)
            agent.remember(state, action, reward, state_, done)
            q_loss = agent.learn()
            score+=reward
            state = state_
           
        score_history.append(score)
        q_loss_history.append(q_loss)
        avg_score = np.mean(score_history[-100:])
        
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()
        
        print(f"episode {i} score {score:0.2f} average_score {avg_score:.1f}")

    x = [i+1 for i in range(n_games)]
        
    plot_learning_curve(x, score_history, score_history_filename)                 
    plot_q_curve(x, q_loss_history, qloss_history_filename)