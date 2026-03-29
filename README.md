# Multi-Room Hotel Pricing với thuật toán PPO

Dự án này sử dụng Học tăng cường sâu (Deep Reinforcement Learning), cụ thể là thuật toán **Proximal Policy Optimization (PPO)** kết hợp với mạng **Actor-Critic (Squashed Gaussian Continuous Action)**, để giải quyết bài toán định giá phòng khách sạn (Yield Management) tối đa hóa doanh thu (Revenue). Hệ thống có khả năng định lượng độ co giãn của cầu theo giá (Price Elasticity) và xử lý cùng lúc nhiều loại phòng khác nhau dưới áp lực sức chứa (Capacity Constraints).

## 🗂 Cấu trúc dự án

```text
ppo_hotel_v2/
├── algorithms/       # Core thuật toán RL
│   └── ppo.py        # Triển khai thuật toán PPO-Clip
├── data/             # Dữ liệu & Pipeline xử lý
│   ├── load_data.py  # Xử lý, nội suy thuộc tính và làm sạch dữ liệu
│   └── ...           # Các file CSV chứa dữ liệu khách sạn & Google Trends
├── envs/             # Môi trường tương tác cho Agent
│   └── multi_room_env.py # OpenAI Gym-style Environment giả lập Booking
├── evaluation/       # Phân tích và chấm điểm sau kỷ nguyên huyến luyện
│   ├── evaluate_model.py # Xây dựng các Baseline (Fixed, Random, Demand Heuristics)
│   └── visualize.py  # Vẽ biểu đồ kết quả đánh giá phân tích
├── models/           # Kiến trúc Mạng Neural
│   ├── actor_critic.py # Mạng Backbone dùng chung (Shared) và 2 nhánh Actor/Critic
│   └── demand_models.py # Các mô hình hồi quy Logistic đoán xác suất khách đặt phòng
├── trainers/         # Quy trình huấn luyện
│   └── trainer.py    # Vòng lặp Rollout (512 steps) và Update PPO
├── checkpoints/      # Directory sinh ra tự động lưu .pth file cho Model tốt nhất
├── evaluation_results/ # Directory lưu các file biểu đồ và thống kê CSV sau khi Test
├── main.py           # File chính thực thi vòng huấn luyện (Train)
└── run_evaluation.py # File thực thi đánh giá mô hình (Evaluate)
```

## 🚀 Hướng dẫn chạy dự án

Bạn có thể tương tác với dự án dễ dàng trên Terminal bằng các đoạn Bash cơ bản sau.

### 1. Chuẩn bị Môi trường (Virtual Environment)
Đảm bảo bạn đã kích hoạt môi trường ảo `.venv` và cài đặt những thư viện thiết yếu (như `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`).

*Dành cho Windows (PowerShell):*
```bash
.\.venv\Scripts\activate
```

### 2. Khởi động Huấn luyện (Training)
Lệnh này sẽ load dữ liệu khách sạn truyền vào, tự động học các đặc điểm cung-cầu, khởi tạo môi trường và train thuật toán PPO. 

Ví dụ với `city_hotel_data.csv` (Mặc định lặp 300 vòng):
```bash
python main.py --hotel data/city_hotel_data.csv --iterations 300
```
*(Bạn có thể thay `--hotel data/resort_hotel_data.csv` để đối chiếu với cấu trúc giá khách sạn ngoại ô)*.

### 3. Khởi động Đánh giá (Evaluation)
Sau khi huấn luyện xong, mô hình có điểm `Reward` cao nhất sẽ tự động lưu vào file `checkpoints/best_model.pth`. Bạn có thể dùng model đó trực tiếp đấu lại các phương pháp định giá cố định/ngẫu nhiên (Baseline):

Ví dụ đánh giá qua 50 tập (episodes):
```bash
python run_evaluation.py --hotel data/city_hotel_data.csv --episodes 50
```
Trong quá trình chạy, các số liệu như tỷ lệ Doanh thu tăng lên, đồ thị giá, độ tin cậy P-value đều sẽ tự in ra và kết xuất file thống kê vào thư mục `evaluation_results/`.

### 4. Bỏ qua Huấn luyện - Chạy ngầm Baselines 
Nếu bạn chỉ muốn đối chiếu mức giá cố định (chưa có AI) mà không muốn Load AI:
```bash
python main.py --hotel data/city_hotel_data.csv --baselines
```

*(Ghi chú: Mô hình tự động tận dụng CUDA/GPU nếu máy bạn có Card rời tương thích).*
