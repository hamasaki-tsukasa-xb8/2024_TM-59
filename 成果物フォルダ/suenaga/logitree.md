graph TD
    A[reinforce_main関数開始] --> B[サイトとプレイヤーのリストを定義]
    B --> C[make_similality_matrix関数を呼び出し類似率行列を生成]
    C --> D[make_tendency_matrix関数を呼び出しサイト遷移確率行列を生成]
    D --> E[reward_tableを計算 (類似率行列 × サイト遷移確率行列)]
    E --> F[RecommenderAgentインスタンスをプレイヤーごとに作成]
    F --> G{learning_num > 101か?}
    G -->|Yes| H[save_locale = "many_learning_results"]
    G -->|No| I[save_locale = "few_learning_results"]
    H --> J[学習パラメータを文字列として保存 (save_para)]
    I --> J
    J --> K[シミュレーション開始 (各プレイヤーごとに学習)]
    K --> L[エピソードループ (1000回)]
    L --> M[ランダムに開始サイトを選択]
    M --> N[エージェントが次のサイトを推薦]
    N --> O[Qテーブルを更新 (update_q_table)]
    O --> P[次の状態に遷移]
    P -->|エピソード終了| Q[学習後のQテーブルを正規化]
    Q --> R[reward_tableと比較してプロットを作成]
    R --> S[プロットをファイルに保存]
    S --> T[次のプレイヤーへ]
    T -->|全プレイヤー終了| U[reinforce_main関数終了]

    subgraph parameters.py
        D --> D1[make_tendency_matrix関数開始]
        D1 --> D2[サイト遷移確率行列を初期化]
        D2 --> D3[高確率の遷移先を設定 (high_prob関数)]
        D3 --> D4[他の遷移先の確率をDirichlet分布で設定]
        D4 --> D5[確率を正規化し、負の値をクリップ]
        D5 --> D6[サイト遷移確率行列を返す]
        C --> C1[make_similality_matrix関数開始]
        C1 --> C2[類似率行列を生成]
        C2 --> C3[類似率行列を返す]
    end