import numpy as np
import random

class RecommenderAgent:
    def __init__(self, sites, alpha=0.1, gamma=0.9, epsilon=0.5):
        self.sites = sites
        self.alpha = alpha  # 学習率
        self.gamma = gamma  # 割引率
        self.epsilon = epsilon  # ε-greedy戦略
        self.q_table = np.zeros((len(sites), len(sites)))  # Qテーブル

    def get_state_index(self, site):
        return self.sites.index(site)

    def choose_action(self, state_index):
        # ε-greedy戦略で次の行動を選択
        if random.uniform(0, 1) < self.epsilon:
            # ランダムに行動選択
            print("ランダムに行動選択しました")
            return random.choice(range(len(self.sites)))
        else:
            # Qテーブルに基づいて最適な行動選択
            print("Qテーブルに基づいて最適な行動選択しました")
            return np.argmax(self.q_table[state_index])

    def update_q_table(self, state_index, action_index, reward, next_state_index):
        # Q値の更新
        best_next_q = np.max(self.q_table[next_state_index])
        self.q_table[state_index, action_index] += self.alpha * (reward + self.gamma * best_next_q - self.q_table[state_index, action_index])

    def recommend(self, current_site):
        # 現在のサイトに基づいて次に推薦するサイトを選択
        state_index = self.get_state_index(current_site)
        action_index = self.choose_action(state_index)
        return self.sites[action_index]

# サイトのリスト（例）
sites = ["SiteA", "SiteB", "SiteC", "SiteD", "SiteE"]
current_site = random.choice(sites)  # ランダムに開始サイトを選択
print('開始サイトをランダムに選択;', current_site)
print("\n")

# エージェントのインスタンスを作成
agent = RecommenderAgent(sites)

# シミュレーション（例）
for episode in range(1000):  # 1000回の学習を行う
    #current_site = random.choice(sites)  # ランダムに開始サイトを選択
    
    next_site = agent.recommend(current_site)
    print(f"Current Site: {current_site}, Recommended Site: {next_site}")

    # print("現在のQテーブルを表示:")
    # print(agent.q_table)

    while True:
        user_input = input("次に遷移するサイトを入力してください (終了するには 'exit' と入力): ")
        if user_input in sites:
            break
        # if user_input in sites:
        #     recommended_site = agent.recommend(user_input)
        #     print(f"Recommended Site: {recommended_site}")
        elif user_input == "exit":
            print("終了します。")
            break
        else:
            print("無効なサイトです。再度入力してください。")
    
    if user_input == "exit":
        break
    

    current_site = user_input

    #for step in range(10):  # 各エピソードで最大10ステップ
    # 次のサイトを推薦

    
    # ユーザーが次のサイトをクリックするかどうかをシミュレート（仮に50%の確率でクリックする）***
    #reward = 1 if random.random() < 0.5 else 0
    if current_site == next_site:
        reward = 1
        print('recommendation is correct')
    else:
        reward = 0
        print('recommendation is not correct')
    
    # 状態遷移とQテーブルの更新
    current_site_index = agent.get_state_index(current_site)
    next_site_index = agent.get_state_index(next_site)
    action_index = agent.get_state_index(next_site)
    
    # Q値を更新
    agent.update_q_table(current_site_index, action_index, reward, next_site_index)
    
    # 次の状態に遷移
    current_site = next_site
    print("学習後のQテーブル:")
    print(agent.q_table)
    print("\n")     
    print("次の行動をrecommendします") 
