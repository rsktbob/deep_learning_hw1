# Grid World — Reinforcement Learning Policy Evaluator

互動式強化學習網格世界，支援策略評估（Policy Evaluation）與價值迭代（Value Iteration）。提供兩種使用方式：Flask 後端版與純前端版。

---

## 專案結構

```
hw1/
├── app.py                  # Flask 後端（API + 路由）
├── requirements.txt        # Python 相依套件
├── README.md
└── templates/
    ├── index.html          # Flask 版前端（需搭配後端）
    └── standalone.html     # 純前端版（無需任何後端）
```

---

## 作業對應說明

| 作業 | 內容 | 實作位置 |
|------|------|----------|
| HW1-1 | n×n 網格地圖，可設定起始點、終點、障礙物 | `index.html` / `standalone.html` |
| HW1-2 | 隨機生成策略（箭頭）+ 策略評估 V(s) | `app.py: policy_evaluation()` / `standalone.html: policyEvaluation()` |
| HW1-3 | 價值迭代找出最佳策略 + 顯示最佳路徑 | `app.py: value_iteration()` / `standalone.html: valueIteration()` |

---

## 方式一：Flask 版（推薦）

### 環境需求

- Python 3.8+
- Flask 3.0+

### 安裝與啟動

```bash
# 1. 安裝相依套件
pip install -r requirements.txt

# 2. 啟動伺服器
python app.py

# 3. 開啟瀏覽器
#    主頁面（Flask 版）：
http://localhost:5000/

#    純前端版：
點擊專案中deep_learning_hw1/index.html
```

### API 端點

| 方法 | 路徑 | 說明 |
|------|------|------|
| `GET` | `/` | 主頁面 |
| `GET` | `/standalone` | 純前端版頁面 |
| `POST` | `/api/random_policy` | 產生隨機策略（箭頭） |
| `POST` | `/api/evaluate` | 策略評估 + 價值迭代 |

#### `POST /api/random_policy`

**Request Body：**
```json
{
  "n": 5,
  "obstacles": [[1, 2], [3, 4]],
  "goal": [4, 4]
}
```

**Response：**
```json
{
  "rand_policy": {
    "0,0": "↓",
    "0,1": "→",
    ...
  }
}
```

#### `POST /api/evaluate`

**Request Body：**
```json
{
  "n": 5,
  "obstacles": [[1, 2]],
  "start": [0, 0],
  "goal": [4, 4],
  "rand_policy": { "0,0": "↓", ... }
}
```

**Response：**
```json
{
  "rand_values":  { "0,0": 0.327, ... },
  "opt_policy":   { "0,0": "↓",  ... },
  "opt_values":   { "0,0": 0.531, ... },
  "path":         [[0,0], [1,0], ..., [4,4]]
}
```

---

## 方式二：純前端版（無需後端）

直接用瀏覽器開啟 `deep_learning_hw1/index.html`，不需要安裝任何套件或啟動伺服器。

所有 RL 演算法（Policy Evaluation、Value Iteration）皆以 JavaScript 實作，完全在瀏覽器中執行。

---

## 操作流程

### 步驟一：設定地圖

1. 選擇網格大小（5 ~ 9）
2. 點選「**🟢 起始點**」模式，點擊格子設定起始位置（綠色）
3. 點選「**🔴 目標點**」模式，點擊格子設定目標位置（紅色）
4. 點選「**⬛ 障礙物**」模式，點擊格子設定障礙物（灰色，最多 n-2 個）
5. 點選「**✕ 清除**」模式可取消任意格子的設定

### 步驟二：產生隨機策略

點擊「**🎲 產生隨機策略**」，上方 HW1-2 的網格會立即顯示每個格子的隨機方向箭頭。
此時箭頭**固定不再改變**，作為 HW1-2 的策略評估基礎。

### 步驟三：執行評估

點擊「**▶ 評估策略 + 價值迭代**」，同時執行兩件事：

- **HW1-2 網格**：對固定的隨機策略執行策略評估，每格顯示 V(s) 數值與熱力圖
- **HW1-3 網格**：執行價值迭代，找出最佳策略箭頭與 V\*(s)，並以黃色標示最佳路徑

---

## 演算法說明

### 獎勵設計

| 事件 | 獎勵 |
|------|------|
| 每走一步 | **−0.1** |
| 到達目標 | **+1.0** |
| 折扣因子 γ | **0.9** |

終點為吸收態（Terminal State），`V(goal) = 0`，獎勵在進入目標時才給予。

**範例：**
```
目標鄰格  V = 1.0 + 0.9 × 0   = 1.000
兩步之外  V = −0.1 + 0.9 × 1.0 = 0.800
三步之外  V = −0.1 + 0.9 × 0.8 = 0.620
```

### HW1-2：策略評估（Policy Evaluation）

對給定的固定策略 π，反覆套用 Bellman Expectation 方程直到收斂：

$$V(s) \leftarrow R(s, \pi(s)) + \gamma \cdot V(s')$$

- 收斂條件：所有狀態的更新量 δ < θ = 10⁻⁶
- 最大迭代次數：10,000

### HW1-3：價值迭代（Value Iteration）

不依賴固定策略，對每個狀態取所有動作中的最大 Q 值：

$$V(s) \leftarrow \max_{a} \left[ R(s,a) + \gamma \cdot V(s') \right]$$

收斂後從 V\* 反推最佳策略：

$$\pi^*(s) = \arg\max_{a} \left[ R(s,a) + \gamma \cdot V^*(s') \right]$$

最佳路徑：從起始點沿 π\* 逐步走到終點。

---

## 視覺說明

| 顏色 | 意義 |
|------|------|
| 🟩 綠色格子 | 起始點 |
| 🟥 紅色格子 | 目標點（顯示 ★） |
| ⬛ 灰色格子 | 障礙物（斜線紋路） |
| 🟡 黃色邊框 | HW1-3 最佳路徑 |
| 綠色背景 | 高 V(s) 值 |
| 紅色背景 | 低 V(s) 值 |
| ↑ ↓ ← → | 該格的策略方向 |
