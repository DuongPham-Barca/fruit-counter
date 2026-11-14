# TÀI LIỆU TỔNG HỢP DỰ ÁN FRUIT COUNTER

> Tài liệu kỹ thuật đầy đủ về kiến trúc, dữ liệu, huấn luyện, đánh giá, theo dõi, đếm
> trái cây và cách vận hành toàn bộ repository.

## Mục lục

1. [Tổng quan bài toán](#1-tổng-quan-bài-toán)
2. [Phạm vi và đầu ra](#2-phạm-vi-và-đầu-ra)
3. [Cấu trúc repository](#3-cấu-trúc-repository)
4. [Luồng xử lý tổng thể](#4-luồng-xử-lý-tổng-thể)
5. [Kiến trúc FruitCounterNet](#5-kiến-trúc-fruitcounternet)
6. [Kiến thức nền tảng CNN](#6-kiến-thức-nền-tảng-cnn)
7. [Dataset và cách chia dữ liệu](#7-dataset-và-cách-chia-dữ-liệu)
8. [Quy trình huấn luyện](#8-quy-trình-huấn-luyện)
9. [Loss function](#9-loss-function)
10. [Optimizer và learning-rate scheduler](#10-optimizer-và-learning-rate-scheduler)
11. [Đánh giá mô hình](#11-đánh-giá-mô-hình)
12. [TensorBoard](#12-tensorboard)
13. [Checkpoint và resume](#13-checkpoint-và-resume)
14. [Inference, tracking và đếm](#14-inference-tracking-và-đếm)
15. [Cài đặt và các lệnh sử dụng](#15-cài-đặt-và-các-lệnh-sử-dụng)
16. [Output của mỗi lần chạy](#16-output-của-mỗi-lần-chạy)
17. [Kiểm thử](#17-kiểm-thử)
18. [Lỗi thường gặp](#18-lỗi-thường-gặp)
19. [Giới hạn hiện tại và hướng phát triển](#19-giới-hạn-hiện-tại-và-hướng-phát-triển)
20. [Thứ tự đọc source code](#20-thứ-tự-đọc-source-code)

---

## 1. Tổng quan bài toán

Dự án nhận đầu vào là video quay sẵn của một dây chuyền trái cây và thực hiện ba nhiệm vụ:

1. Nhận diện từng quả cam hoặc táo trong mỗi frame.
2. Theo dõi mỗi quả qua nhiều frame bằng một track ID.
3. Đếm đúng một lần khi tâm của track đi qua đường đếm theo chiều đã cấu hình.

Đây không chỉ là bài toán classification. Classification chỉ trả lời toàn ảnh thuộc lớp
nào, còn dự án phải xác định đồng thời:

- Trái cây nằm ở đâu: bounding box `(x1, y1, x2, y2)`.
- Trái cây thuộc lớp nào: `orange` hoặc `apple`.
- Độ tin cậy của dự đoán.
- Đối tượng ở frame hiện tại tương ứng với đối tượng nào ở frame trước.
- Đối tượng đã đi qua đường đếm hay chưa.

Pipeline chính:

```text
Video đầu vào
    ↓
FFmpeg decode từng frame
    ↓
FruitCounterNet phát hiện bounding box + class + confidence
    ↓
ByteTrack-style tracker gán track ID
    ↓
LineCounter kiểm tra hướng và sự kiện qua đường
    ↓
Video có annotation + summary.json + events.csv
```

---

## 2. Phạm vi và đầu ra

### 2.1. Lớp nhận diện

| Dataset class ID | Model label | Tên lớp |
| --- | --- | --- |
| Không áp dụng | 0 | background |
| 0 | 1 | orange |
| 1 | 2 | apple |

Faster R-CNN luôn dành model label `0` cho background. Vì vậy dataset class `0` phải
được chuyển thành model label `1`, và dataset class `1` thành model label `2`.

### 2.2. Đầu ra khi đếm video

Một lần chạy video tạo:

- `annotated.mp4`: video có bounding box, class, track ID, đường đếm và tổng số.
- `summary.json`: metadata video, cấu hình đường đếm, tổng số cam/táo và danh sách event.
- `events.csv`: mỗi dòng là một lần đếm, gồm frame, thời gian, track ID và class.

### 2.3. Yêu cầu quan trọng

- Camera nên cố định.
- Hướng di chuyển chính phải được xác định: trái sang phải (`ltr`) hoặc phải sang trái (`rtl`).
- Đường đếm phải nằm trong vùng đối tượng được theo dõi ổn định.
- Dữ liệu thật phải giống góc camera, ánh sáng và nền khi triển khai.
- Model detector và backbone được khởi tạo ngẫu nhiên, không sử dụng pretrained weights.

---

## 3. Cấu trúc repository

```text
fruit-counter/
├── pyproject.toml
│   └── Metadata package, dependencies, console command và Ruff config.
├── README.md
│   └── Hướng dẫn nhanh.
├── TAI_LIEU_DU_AN.md
│   └── Tài liệu đầy đủ này.
├── PLAN.md
│   └── Phạm vi, yêu cầu và lộ trình dự án.
├── data/
│   ├── README.md                Quy tắc dữ liệu video thật.
│   ├── cache/                   Annotation cache, không commit.
│   └── datasets/                Dataset đã chuẩn bị, không commit.
├── src/fruit_counter/
│   ├── __main__.py              CLI `fruit-counter`.
│   ├── labels.py                Mapping class dataset ↔ model.
│   ├── dataset.py               YOLO parser, Dataset, augmentation và DataLoader.
│   ├── model.py                 CNN, ResNet, FPN, RPN, RoI box head.
│   ├── train.py                 Config, checkpoint, schedule và public training API.
│   ├── engine.py                Train một epoch, validate, TensorBoard và checkpoint.
│   ├── evaluate.py              Loss, AP, precision, recall và confusion matrix.
│   ├── fruit_counter_net.py     Load checkpoint và inference trên frame.
│   ├── tracker.py               Kalman, Hungarian matching, line counter.
│   ├── pipeline.py              Pipeline đếm video end-to-end.
│   ├── video.py                 FFmpeg/FFprobe I/O.
│   ├── demo.py                  Sinh video giả lập.
│   ├── prepare_coco.py          Logic tải/chuyển COCO.
└── tests/
    ├── test_model.py
    ├── test_data_and_metrics.py
    ├── test_train.py
    ├── test_tracker.py
    └── test_pipeline.py
```

### 3.1. Các file nên đọc đầu tiên

Nếu mới tiếp cận repository, đọc theo thứ tự:

1. `src/fruit_counter/__main__.py` để hiểu các lệnh CLI và tham số.
2. `src/fruit_counter/dataset.py` để hiểu dữ liệu đầu vào.
3. `src/fruit_counter/model.py` để hiểu architecture.
4. `src/fruit_counter/train.py` để hiểu config, schedule và checkpoint.
5. `src/fruit_counter/engine.py` để hiểu training step, validation và logging.
6. `src/fruit_counter/tracker.py` để hiểu theo dõi và đếm.
7. `src/fruit_counter/pipeline.py` để hiểu luồng video hoàn chỉnh.

---

## 4. Luồng xử lý tổng thể

### 4.1. Luồng huấn luyện

```text
YOLO images + labels
    ↓
YoloDetectionDataset
    ↓
Train augmentation / Validation transform
    ↓
DataLoader
    ↓
Faster R-CNN forward
    ↓
RPN losses + RoI classification/regression losses
    ↓
Gradient accumulation + AMP
    ↓
Gradient clipping
    ↓
SGD optimizer step
    ↓
Warmup + cosine learning-rate schedule
    ↓
Validation loss + detection metrics
    ↓
TensorBoard + JSON + checkpoints
```

### 4.2. Luồng inference video

```text
Input MP4
    ↓ ffprobe
Đọc width, height, FPS
    ↓ ffmpeg
Decode và resize RGB frame
    ↓
FruitCounterNetDetector
    ↓
Cross-class NMS
    ↓
Detection(label, bbox, score)
    ↓
Kalman prediction + Hungarian IoU matching
    ↓
Track ID + class confidence vote
    ↓
Kiểm tra tâm track qua line
    ↓
Annotation + MP4/JSON/CSV output
```

---

## 5. Kiến trúc FruitCounterNet

FruitCounterNet là tên mô hình của dự án, được xây dựng dựa trên thiết kế
hai giai đoạn Faster R-CNN với backbone, FPN, RPN và box head do dự án khai báo.

Model được khai báo tường minh trong `src/fruit_counter/model.py`.

```text
RGB image
    ↓
ResNet CNN backbone
    ↓
Feature Pyramid Network (FPN)
    ↓
Region Proposal Network (RPN)
    ↓
Proposals / candidate boxes
    ↓
MultiScaleRoIAlign 7×7
    ↓
4 × (Conv2d → GroupNorm → ReLU)
    ↓
Flatten → Linear(1024) → ReLU
    ├── Class logits: background/orange/apple
    └── Box deltas: 4 giá trị cho mỗi class
```

### 5.1. Backbone ResNet

Repository hỗ trợ:

| Backbone | Residual block | Số block mỗi stage | Output channels |
| --- | --- | --- | --- |
| ResNet-18 | BasicBlock | 2, 2, 2, 2 | 64, 128, 256, 512 |
| ResNet-34 | BasicBlock | 3, 4, 6, 3 | 64, 128, 256, 512 |
| ResNet-50 | Bottleneck | 3, 4, 6, 3 | 256, 512, 1024, 2048 |

Stem của ResNet:

```text
Conv2d(3 → 64, kernel 7×7, stride 2)
→ GroupNorm
→ ReLU
→ MaxPool 3×3 stride 2
```

`BasicBlock`:

```text
Conv3×3 → GroupNorm → ReLU → Conv3×3 → GroupNorm
              + residual/identity connection
→ ReLU
```

`Bottleneck`:

```text
Conv1×1 → GN → ReLU
→ Conv3×3 → GN → ReLU
→ Conv1×1 → GN
              + residual/identity connection
→ ReLU
```

Residual connection giúp gradient truyền qua mạng sâu ổn định hơn và giảm hiện tượng
vanishing gradient.

### 5.2. Feature Pyramid Network

FPN nhận feature maps từ bốn stage của ResNet và tạo các mức đặc trưng P2–P6 có cùng
số channel nhưng khác độ phân giải.

- Feature map độ phân giải cao hỗ trợ quả nhỏ.
- Feature map độ phân giải thấp có receptive field lớn, hỗ trợ quả lớn.
- `fpn_channels` mặc định là 256.

### 5.3. Anchor Generator

Anchor là các bounding box mẫu được đặt tại mỗi vị trí feature map.

Mặc định:

- Anchor sizes: `16, 32, 64, 128, 256` pixel.
- Aspect ratios: `0.75, 1.0, 1.33`.
- Có năm mức tương ứng P2–P6.

Trái cây gần tròn nên tỷ lệ anchor tập trung quanh 1:1.

### 5.4. Region Proposal Network

RPN head gồm hai tầng:

```text
Conv3×3 → ReLU → Conv3×3 → ReLU
```

Sau đó tách thành:

- `objectness_logits`: anchor có chứa object hay background.
- `box_deltas`: điều chỉnh anchor gần bounding box thật.

RPN trả logits, không gọi sigmoid trực tiếp trong model head. Loss
`BCEWithLogitsLoss` đã kết hợp sigmoid và binary cross entropy theo cách ổn định số học.

### 5.5. RoIAlign và box head

`MultiScaleRoIAlign` lấy vùng proposal từ FPN và đưa mọi vùng về kích thước 7×7.

Box head:

```text
4 × (Conv3×3, 256 channels → GroupNorm → ReLU)
→ Flatten
→ Linear(12544 → 1024)
→ ReLU
```

Predictor cuối trả:

- `class_logits`: 3 logits cho background, orange, apple.
- `box_deltas`: 12 số, tương ứng 4 tọa độ × 3 class.

Không thêm Softmax trực tiếp vào head vì CrossEntropyLoss cần raw logits. Khi inference,
TorchVision Faster R-CNN áp dụng Softmax để tạo class probabilities.

---

## 6. Kiến thức nền tảng CNN

### 6.1. Convolution

Convolution dùng kernel trượt trên ảnh hoặc feature map để học cạnh, texture, màu sắc
và cấu trúc không gian. Với input nhiều channel, mỗi output channel là tổng convolution
trên tất cả input channel cộng bias.

Các tham số chính:

- `kernel_size`: kích thước cửa sổ, thường là 3×3.
- `stride`: bước nhảy; stride 2 làm giảm một nửa kích thước không gian.
- `padding`: thêm biên; padding 1 giữ kích thước với kernel 3×3.
- `channels`: số feature maps mô hình học.

### 6.2. ReLU

ReLU được định nghĩa:

```text
ReLU(x) = max(0, x)
```

ReLU tạo phi tuyến, giúp mạng học quan hệ phức tạp. Nếu chỉ xếp nhiều convolution tuyến
tính mà không có activation thì toàn bộ chuỗi vẫn tương đương một phép biến đổi tuyến tính.

### 6.3. Group Normalization

BatchNorm phụ thuộc mean/variance của batch. Object detection trên GPU 4 GB thường chỉ
dùng batch 1–2, làm thống kê BatchNorm không ổn định.

GroupNorm chia channel thành các group và chuẩn hóa trong từng sample, nên ít phụ thuộc
batch size. Dự án dùng GroupNorm trong backbone, FPN và box head.

### 6.4. Sigmoid và Softmax

Sigmoid:

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

Phù hợp với binary classification hoặc multi-label khi các lớp độc lập.

Softmax:

```text
softmax(z_i) = exp(z_i) / Σ_j exp(z_j)
```

Phù hợp với multi-class classification khi chỉ có một lớp đúng. Trong dự án, một
proposal chỉ có thể là background, orange hoặc apple, nên class head dùng Softmax thông
qua CrossEntropyLoss thay vì nhiều sigmoid độc lập.

RPN chỉ hỏi object/background nên dùng BCE-with-logits, tức sigmoid được tích hợp trong loss.

### 6.5. Logits

Logit là output chưa chuẩn hóa của neural network. Giữ logits đến khi tính loss giúp:

- Ổn định số học.
- Tránh tính log của probability rất nhỏ.
- Cho phép `CrossEntropyLoss` và `BCEWithLogitsLoss` kết hợp activation + loss tối ưu.

### 6.6. Random initialization

Mọi convolution được khởi tạo ngẫu nhiên. CNN dùng Kaiming initialization phù hợp với
ReLU. Normalization scale được đặt 1 và bias đặt 0. Nhánh residual cuối được zero-init
để block bắt đầu gần identity mapping.

Không có pretrained ImageNet hoặc pretrained detector. Điều này tuân thủ yêu cầu dự án
nhưng cần nhiều dữ liệu và thời gian train hơn so với fine-tuning pretrained model.

---

## 7. Dataset và cách chia dữ liệu

### 7.1. Cấu trúc dữ liệu YOLO

`YoloDetectionDataset` đọc dữ liệu theo cấu trúc:

```text
data/datasets/coco-fruit/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Mỗi ảnh có một file `.txt` cùng tên trong thư mục `labels`. Mỗi dòng có dạng:

```text
class_id center_x center_y width height
```

Ví dụ:

```text
0 0.500000 0.420000 0.180000 0.240000
1 0.250000 0.610000 0.120000 0.160000
```

Tọa độ đều được chuẩn hóa về `[0, 1]`. Code chuyển chúng thành bounding box tuyệt đối
`(x1, y1, x2, y2)` trước khi đưa vào Faster R-CNN. File label không tồn tại được hiểu là
ảnh background, tức ảnh không có object. Box nhỏ hơn 1 pixel bị bỏ qua.

Mapping nhãn phải được giữ thống nhất:

```text
YOLO 0 (orange) → Faster R-CNN 1
YOLO 1 (apple)  → Faster R-CNN 2
Faster R-CNN 0  → background, không xuất hiện trong file YOLO
```

### 7.2. Dataset hiện có trong repository

Tại thời điểm tài liệu này được tạo, dataset COCO-derived hiện tại có:

| Split | Số ảnh | Tỷ lệ trên tổng 2.602 ảnh |
| --- | ---: | ---: |
| Train | 2.231 | 85,7% |
| Validation | 247 | 9,5% |
| Test | 124 | 4,8% |

Kích thước dataset đã xử lý khoảng 426 MB; cache COCO khoảng 468 MB. Đây là dữ liệu
khởi đầu lấy từ COCO, chưa phải dữ liệu dây chuyền thực tế. Số lượng có thể thay đổi nếu
chạy lại bước chuẩn bị dataset.

### 7.3. Chia 80% train và 20% validation

Quy ước 80/20 thường phù hợp khi chỉ cần train và validation. Để tạo dataset mới với
20% validation:

```sh
fruit-counter prepare-coco \
  --output data/datasets/coco-fruit-80-20 \
  --val-fraction 0.2
```

Nên dùng một thư mục output mới vì ghi đè lên split cũ có thể để lại file thừa. Sau đó
train với:

```sh
fruit-counter train \
  --data data/datasets/coco-fruit-80-20 \
  --output outputs/fruit_counter_net_r50_80_20
```

Lưu ý: tool lấy COCO `train2017`, tách một phần làm validation và dùng ảnh liên quan từ
COCO `val2017` làm test. Vì vậy tỷ lệ cuối trên cả ba split không đúng chính xác 80/20;
`--val-fraction 0.2` là tỷ lệ tách train/val từ nguồn train.

Với video dây chuyền thực tế, không nên random từng frame rồi chia. Các frame liền nhau
rất giống nhau và sẽ gây data leakage. Cách chuẩn hơn là chia theo video hoặc phiên quay:

- 80% phiên quay cho train.
- 20% phiên quay khác hẳn cho validation.
- Nếu có đủ dữ liệu, giữ riêng một test set chưa từng dùng để chỉnh tham số.

### 7.4. Augmentation

Chỉ train split có augmentation ngẫu nhiên:

- Horizontal flip với xác suất 0,5.
- Translate tối đa 10% và scale trong khoảng 0,7–1,3 với xác suất 0,5.
- Color jitter: brightness, contrast, saturation và hue.
- Clamp box vào biên ảnh và loại box nhỏ hơn 2 pixel sau biến đổi.
- Chuyển pixel về `float32` trong khoảng `[0, 1]`.

Validation và test chỉ chuyển ảnh thành tensor, không có augmentation ngẫu nhiên. Đây là
điều bắt buộc để metric giữa các epoch có thể so sánh công bằng.

### 7.5. DataLoader

Object detection có số box trên mỗi ảnh khác nhau, nên không thể stack target như bài
toán classification thông thường. Hàm `collate` trả về list ảnh và list target. Train
loader shuffle dữ liệu; validation loader không shuffle. Khi chạy CUDA, loader bật
`pin_memory` để tăng tốc copy dữ liệu sang GPU.

---

## 8. Quy trình huấn luyện

### 8.1. Trình tự một lần train

Luồng trong `__main__.py`, `train.py` và `engine.py`:

1. Đọc tham số CLI và tạo `ModelConfig`, `TrainConfig`.
2. Kiểm tra CUDA, các giá trị cấu hình và output directory.
3. Cố định seed cho Python, NumPy, PyTorch và CUDA.
4. Khởi tạo Faster R-CNN từ trọng số ngẫu nhiên.
5. Tạo SGD optimizer, AMP gradient scaler và DataLoader.
6. Nếu resume, khôi phục model, optimizer, scaler, epoch, step và RNG state.
7. Với mỗi epoch: forward → tổng loss → backward → clip gradient → optimizer step.
8. Chạy validation loss và detection metrics.
9. Ghi TensorBoard, JSON, ảnh dự đoán và confusion matrix.
10. Lưu `last.pt`, đồng thời cập nhật best checkpoint khi metric tốt hơn.

### 8.2. Cấu hình mặc định của lệnh train

Giá trị mặc định của `fruit-counter train` (xem `fruit-counter train --help`):

| Tham số | Giá trị |
| --- | --- |
| Backbone | ResNet-50 |
| Epoch | 100 |
| Batch size thực trên một forward | 2 |
| Gradient accumulation | 4 batch |
| Effective batch size | 8 ảnh |
| Learning rate | 0,01 |
| Optimizer | SGD |
| Momentum | 0,9 |
| Weight decay | `1e-4` |
| Worker | 4 |
| Device | CUDA |
| Mixed precision | Bật |
| Seed | 0 |

Mọi giá trị trên đổi được bằng tham số CLI. Learning rate nếu không truyền sẽ tính theo
`0,02 × effective batch / 16` (với batch 8 ra đúng 0,01). Cấu hình nội bộ còn
có warmup 1.000 step, cosine decay, minimum LR ratio 0,01, gradient clipping 10 và
validation sau mỗi epoch.

### 8.3. Gradient accumulation

Với `batch_size=2` và `accumulate=4`, code chạy bốn forward/backward nhỏ rồi mới gọi
optimizer một lần:

```text
effective batch = batch_size × accumulate = 2 × 4 = 8 ảnh
```

Loss được chia cho `accumulate` trước khi backward để gradient có scale hợp lý. Kỹ thuật
này hữu ích cho GPU ít VRAM, nhưng không hoàn toàn giống batch 8 thật vì mỗi forward
chỉ thấy hai ảnh. GroupNorm được dùng vì ổn định với batch nhỏ hơn BatchNorm.

### 8.4. Automatic Mixed Precision

AMP dùng `torch.autocast` và `GradScaler` để phần lớn phép tính CUDA chạy ở precision
thấp hơn, thường giảm VRAM và tăng tốc. Trước khi clip gradient, code gọi `unscale_`.
Nếu gặp loss `NaN/Inf`, batch đó bị bỏ qua; quá nhiều batch lỗi liên tiếp sẽ dừng train.

Tắt AMP để debug:

```sh
fruit-counter train --data data/datasets/coco-fruit --output outputs/amp_off --no-amp
```

### 8.5. Resize ảnh trong detector

Trong lúc train, cạnh ngắn được chọn ngẫu nhiên từ `416, 448, 480, 512`; cạnh dài bị
giới hạn ở 912 pixel. Multi-scale training giúp mô hình bền hơn với thay đổi kích thước.
Khi inference, TorchVision dùng kích thước xác định từ cấu hình, không random như train.

---

## 9. Loss function

Faster R-CNN trả về bốn thành phần loss:

```text
loss_total = loss_objectness
           + loss_rpn_box_reg
           + loss_classifier
           + loss_box_reg
```

### 9.1. RPN objectness loss

RPN phân loại mỗi anchor thành object hoặc background. Nó dùng binary cross entropy
with logits:

```text
BCEWithLogits(z, y) = -[y log(sigmoid(z)) + (1-y) log(1-sigmoid(z))]
```

Sigmoid đã nằm bên trong loss để ổn định số học. Vì thế không thêm `nn.Sigmoid()` vào
RPN head.

### 9.2. RPN box regression loss

RPN dự đoán bốn delta để biến anchor thành proposal sát object. Sai số tọa độ dùng
Smooth L1, ít nhạy với outlier hơn MSE.

### 9.3. ROI classification loss

Mỗi proposal được phân vào một trong ba lớp `background/orange/apple` bằng cross
entropy. Nếu logits là `z_k`:

```text
softmax(z_k) = exp(z_k) / Σ exp(z_j)
cross_entropy = -log(probability của lớp đúng)
```

Các lớp loại trừ lẫn nhau nên dùng Softmax/CrossEntropy, không dùng hai sigmoid độc lập.

### 9.4. ROI box regression loss

Box head dự đoán `3 × 4 = 12` delta tọa độ, bốn giá trị cho mỗi class. Chỉ phần tương
ứng với class mục tiêu được dùng để tính Smooth L1 loss.

### 9.5. Đọc loss đúng cách

- Train loss giảm cho biết mô hình khớp train set tốt dần, nhưng chưa chứng minh khả năng
  tổng quát hóa.
- Validation loss giảm cho biết mô hình hoạt động tốt hơn trên dữ liệu chưa train.
- Train loss tiếp tục giảm nhưng validation loss tăng là dấu hiệu overfitting.
- Detection metric như mAP50 vẫn là tiêu chí chính để chọn detector, vì loss tốt không
  luôn tương đương box và class tốt sau NMS.

---

## 10. Optimizer và learning-rate scheduler

### 10.1. SGD

Project dùng SGD với momentum 0,9 và weight decay `1e-4`. Momentum tích lũy hướng của
gradient qua nhiều step, giúp tối ưu ổn định hơn. Weight decay hạn chế trọng số quá lớn
và đóng vai trò regularization.

Nếu `lr=None` trong API nội bộ, learning rate tự scale theo effective batch:

```text
base_lr = 0,02 × effective_batch / 16
```

Entrypoint hiện đặt rõ `lr=0.01`, phù hợp effective batch 8 theo công thức trên.

### 10.2. Warmup và cosine decay

Trong những step đầu, learning rate tăng tuyến tính từ nhỏ lên base LR. Warmup tránh
gradient lớn làm training mất ổn định khi model còn khởi tạo ngẫu nhiên. Sau warmup,
learning rate giảm theo cosine đến `base_lr × 0,01` ở cuối quá trình.

### 10.3. Gradient clipping

Sau khi unscale gradient, code giới hạn global gradient norm ở 10. TensorBoard ghi
`train/grad_norm`; nếu giá trị thường xuyên rất lớn hoặc xuất hiện batch không finite,
nên giảm learning rate và kiểm tra annotation.

---

## 11. Đánh giá mô hình

### 11.1. IoU

Intersection over Union đo phần giao của hai box:

```text
IoU = diện tích giao / diện tích hợp
```

Dự đoán được xem là match với ground truth nếu IoU ít nhất 0,5. Mỗi ground-truth box
chỉ được match một lần.

### 11.2. Precision và recall

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
```

- Precision cao: ít báo nhầm trái cây.
- Recall cao: ít bỏ sót trái cây.
- Trong project, precision/recall được tính ở confidence threshold 0,5 và IoU 0,5.
- `precision_macro` và `recall_macro` là trung bình không trọng số của orange và apple.

### 11.3. AP50 và mAP50

AP50 là diện tích dưới đường precision–recall của từng class ở IoU 0,5. Tất cả dự đoán
mà model xuất ra được xếp hạng theo confidence để tạo đường cong. `mAP50` là trung bình
AP50 của orange và apple có ground truth.

Checkpoint chính `best_map50.pt` được chọn theo mAP50 cao nhất, không theo train loss.

### 11.4. Confusion matrix cho object detection

Ma trận có hàng là ground truth, cột là prediction:

| Ground truth \ Prediction | background | orange | apple |
| --- | ---: | ---: | ---: |
| background | — | orange báo dư | apple báo dư |
| orange | orange bị bỏ sót | orange đúng | orange bị nhầm apple |
| apple | apple bị bỏ sót | apple bị nhầm orange | apple đúng |

Dự đoán chỉ tham gia confusion matrix khi score ít nhất 0,5. Background row biểu diễn
false positive; background column biểu diễn false negative.

### 11.5. Validation loss

TorchVision detector chỉ trả loss ở training mode. Hàm `evaluate_loss` tạm chuyển model
sang train mode trong `no_grad`, nhưng project dùng GroupNorm và không có Dropout nên
không làm thay đổi running statistics. RNG được cô lập bằng `fork_rng` để validation
không làm thay đổi augmentation/sampling của epoch sau.

Validation hiện chạy hai pass: một pass lấy loss và một pass inference lấy metrics.
Điều này tốn thêm thời gian nhưng giữ logic rõ ràng và metric chính xác.

### 11.6. Validation và test khác nhau

- Validation: chạy trong quá trình train để theo dõi và chọn checkpoint.
- Test: chỉ chạy sau khi đã chọn model và tham số, nhằm báo cáo kết quả cuối.
- Không nên liên tục chỉnh mô hình dựa trên test set, vì khi đó test đã trở thành val.

---

## 12. TensorBoard

Khởi động TensorBoard từ repository root:

```sh
tensorboard --logdir outputs/fruit_counter_net_r50_coco_scratch/logs/tensorboard
```

Sau đó mở địa chỉ TensorBoard in trên terminal, thường là `http://localhost:6006`.

### 12.1. Các scalar đã được lưu

| Nhóm | Tag tiêu biểu | Ý nghĩa |
| --- | --- | --- |
| Train theo step | `train/loss_total` và từng loss | Loss cửa sổ gần nhất |
| Train theo epoch | `epoch/loss_total` và từng loss | Loss trung bình cả epoch |
| Tối ưu | `train/lr`, `train/grad_norm` | Learning rate và gradient norm |
| Hiệu năng | `train/images_per_second` | Tốc độ xử lý |
| Độ ổn định | `train/skipped_batches` | Số batch NaN/Inf bị bỏ |
| Validation loss | `val/loss_total` và bốn loss thành phần | Loss trên val set |
| Detection | `val/mAP50` | mAP tại IoU 0,5 |
| Theo class | `val/AP50_orange`, `val/AP50_apple` | AP50 từng class |
| Precision/Recall | `val/precision_macro`, `val/recall_macro` | Trung bình macro hai class |

TensorBoard còn lưu:

- Text cấu hình ở step 0.
- Ảnh dự đoán trên một số ảnh validation sau mỗi lần eval.
- Confusion matrix dưới dạng image.

VRAM peak và giá trị tốt nhất (`best_map`, `best_val_loss`) chỉ ghi vào
`logs/metrics.jsonl`, không lặp lại trên TensorBoard vì đọc được từ chính các đường cong.

Các dữ liệu trên là đủ tốt cho đồ án. Có thể bổ sung PR curve, histogram trọng số,
histogram gradient hoặc sample lỗi hard-negative, nhưng chúng làm log lớn và chậm hơn;
chỉ nên thêm khi cần phân tích sâu.

---

## 13. Checkpoint và resume

### 13.1. Các file checkpoint

| File | Khi nào cập nhật | Mục đích |
| --- | --- | --- |
| `last.pt` | Cuối mỗi epoch và khi bắt `Ctrl+C` | Resume training |
| `best_map50.pt` | Khi validation mAP50 lập kỷ lục mới | Inference/evaluate, lựa chọn chính |
| `best_val_loss.pt` | Khi validation loss thấp nhất mới | So sánh hoặc phân tích |
| `epoch_NNN.pt` | Khi cấu hình `save_every > 0` | Lưu mốc lịch sử |

Checkpoint được ghi atomically qua file tạm rồi replace, giảm nguy cơ file hỏng nếu tiến
trình dừng đúng lúc đang save.

### 13.2. Nội dung checkpoint

`last.pt` chứa:

- Model weights.
- Optimizer state và momentum.
- AMP GradScaler state.
- Model config, train config và danh sách class.
- Epoch, global optimizer step.
- Best mAP50 và best validation loss.
- RNG state của Python, NumPy, PyTorch, CUDA và DataLoader generator.

Hai best checkpoint loại bỏ optimizer, scaler và RNG để nhẹ hơn và dành cho inference.

### 13.3. Khi bấm Ctrl+C

Code bắt `KeyboardInterrupt` và ghi `last.pt` với trạng thái sau optimizer step gần nhất.
Gradient mới tích lũy nhưng chưa đủ để thực hiện optimizer step sẽ không được lưu. Epoch
đang dang dở cũng sẽ chạy lại khi resume. Vì vậy:

- `last.pt` là trạng thái train mới nhất có thể tiếp tục, không phải model có loss tốt nhất.
- `best_map50.pt` là model có mAP50 validation tốt nhất đã hoàn thành evaluation.
- `best_val_loss.pt` là model có validation loss thấp nhất.

### 13.4. Resume đúng cách

Tiếp tục run mặc định:

```sh
fruit-counter train \
  --data data/datasets/coco-fruit \
  --output outputs/fruit_counter_net_r50_coco_scratch \
  --resume auto
```

Hoặc chỉ rõ checkpoint:

```sh
fruit-counter train \
  --data data/datasets/coco-fruit \
  --output outputs/fruit_counter_net_r50_coco_scratch \
  --resume outputs/fruit_counter_net_r50_coco_scratch/checkpoints/last.pt
```

Khi resume phải dùng lại dataset, batch size, accumulation, LR, optimizer-related config,
seed và model architecture. Code chủ động báo lỗi nếu các giá trị quan trọng thay đổi,
vì thay đổi âm thầm sẽ phá tính liên tục của training.

Lưu ý: `--resume auto` tìm `<output>/checkpoints/last.pt`.

---

## 14. Inference, tracking và đếm

### 14.1. Detector wrapper

`FruitCounterNetDetector` đọc model config và weights từ checkpoint, chuyển ảnh NumPy sang
tensor, chạy model ở inference mode và trả danh sách `Detection(label, bbox, score)`.
Nên dùng `best_map50.pt` cho video thực tế.

### 14.2. Tracker kiểu ByteTrack

Mỗi frame được xử lý theo hai tầng confidence:

1. Kalman filter dự đoán box mới của mọi track theo mô hình vận tốc không đổi.
2. Detection score cao được ghép với track bằng Hungarian assignment trên chi phí
   `1 - IoU`.
3. Detection score thấp được dùng ở vòng ghép thứ hai để cứu track đang che khuất.
4. Detection score cao chưa ghép được có thể tạo tentative track mới.
5. Track đủ số hit được confirmed; track mất quá lâu bị xóa.

Giá trị mặc định trong `LineCounter`:

| Tham số | Giá trị | Ý nghĩa |
| --- | ---: | --- |
| `high_score` | 0,5 | Ngưỡng detection chất lượng cao |
| `low_score` | 0,1 | Ngưỡng thấp nhất còn xét tracking |
| `new_track_score` | 0,6 | Ngưỡng tạo track mới |
| `min_iou` | 0,2 | IoU tối thiểu khi ghép |
| `min_hits` | 2 | Số lần match để confirm |
| `max_missing` | 30 frame | Thời gian giữ track bị mất |

Class của track được quyết định bằng tổng phiếu có trọng số confidence qua các frame,
giảm hiện tượng nhãn orange/apple nhấp nháy.

### 14.3. Logic đếm qua đường

Đường đếm là đường dọc tại:

```text
line_x = chiều_rộng_frame_đã_resize × line_ratio
```

Track chỉ được đếm khi:

- Track đã confirmed.
- Track bắt đầu ở phía trước đường theo hướng cấu hình.
- Tâm track đã chuyển sang phía sau đường.
- Track chưa từng được đếm.

`ltr` nghĩa là trái sang phải; `rtl` nghĩa là phải sang trái. Mỗi track ID chỉ sinh một
event. Khi kết thúc video, event được cập nhật lại theo class vote cuối cùng của track.

### 14.4. Xử lý video

FFprobe đọc width, height, FPS và metadata. FFmpeg decode video thành frame RGB, pipeline
resize theo `--width`, chạy detector/tracker/counter, vẽ annotation rồi encode thành MP4.
Âm thanh đầu vào không được giữ lại.

Lệnh `fruit-counter count` mặc định xử lý width 640. Hàm Python `count_video()` mặc định 320,
nhưng CLI truyền rõ 640. Width nhỏ nhanh hơn nhưng có thể làm mất object nhỏ.

### 14.5. Hạn chế tracking

Tracker không có appearance embedding/ReID. Nếu một object biến mất quá `max_missing`,
quay lại và tạo ID mới, nó có thể được đếm lần nữa. Camera rung mạnh, object giao nhau
lâu, FPS thấp hoặc line nằm gần vùng che khuất cũng làm tăng ID switch.

---

## 15. Cài đặt và các lệnh sử dụng

Tất cả lệnh sau chạy từ repository root.

### 15.1. Yêu cầu hệ thống

- Python 3.10 trở lên.
- PyTorch/TorchVision tương thích.
- FFmpeg và FFprobe có trong `PATH` để xử lý video.
- CUDA được khuyến nghị cho train; CPU phù hợp chủ yếu cho test nhỏ.

### 15.2. Cài môi trường

```sh
source .venv/bin/activate
python -m pip install -e . --no-build-isolation
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ffmpeg -version
```

Nếu cần Ruff:

```sh
python -m pip install -e '.[dev]' --no-build-isolation
```

### 15.3. Chuẩn bị dataset

Mặc định khoảng 90/10 trên nguồn COCO train:

```sh
fruit-counter prepare-coco \
  --output data/datasets/coco-fruit
```

Tạo bản 80/20:

```sh
fruit-counter prepare-coco \
  --output data/datasets/coco-fruit-80-20 \
  --val-fraction 0.2
```

### 15.4. Train

Kích hoạt môi trường, sau đó train bằng CLI:

```sh
source .venv/bin/activate

fruit-counter train \
  --data data/datasets/coco-fruit \
  --output outputs/fruit_counter_net_r50_coco_scratch \
  --backbone resnet50 \
  --epochs 100 \
  --batch-size 2 \
  --accumulate 4 \
  --lr 0.01 \
  --workers 4 \
  --device cuda
```

Xem mọi option:

```sh
fruit-counter train --help
```

### 15.5. Resume

```sh
fruit-counter train \
  --data data/datasets/coco-fruit \
  --output outputs/fruit_counter_net_r50_coco_scratch \
  --backbone resnet50 \
  --epochs 100 \
  --batch-size 2 \
  --accumulate 4 \
  --lr 0.01 \
  --workers 4 \
  --device cuda \
  --resume outputs/fruit_counter_net_r50_coco_scratch/checkpoints/last.pt
```

### 15.6. Evaluate

```sh
fruit-counter evaluate \
  --data data/datasets/coco-fruit \
  --weights outputs/fruit_counter_net_r50_coco_scratch/checkpoints/best_map50.pt \
  --split test \
  --device cuda
```

### 15.7. Đếm video thật

```sh
fruit-counter count \
  --input input/video.mp4 \
  --weights outputs/fruit_counter_net_r50_coco_scratch/checkpoints/best_map50.pt \
  --output outputs/count_video_01 \
  --device cuda \
  --width 640 \
  --line 0.5 \
  --direction ltr \
  --min-score 0.1
```

---

## 16. Output của mỗi lần chạy

### 16.1. Output training

Một work directory có dạng:

```text
outputs/fruit_counter_net_r50_coco_scratch/
├── checkpoints/
│   ├── last.pt
│   ├── best_map50.pt
│   └── best_val_loss.pt
├── evaluation/
│   └── val_epoch_001.json
├── logs/
│   ├── metrics.jsonl
│   └── tensorboard/
└── ... metadata cấu hình run
```

- `metrics.jsonl`: một JSON object cho mỗi epoch, phù hợp parse bằng script.
- `val_epoch_NNN.json`: loss, AP, precision, recall, TP/FP/FN và confusion matrix.
- `tensorboard/`: event files để trực quan hóa.
- Checkpoint: trạng thái model như giải thích ở mục 13.

Project không cho bắt đầu một run mới nếu `last.pt` đã tồn tại trong output directory.
Phải dùng `--resume` hoặc chọn output khác, tránh vô tình ghi đè thí nghiệm.

### 16.2. Output đếm video

```text
outputs/count_video_01/
├── annotated.mp4
├── summary.json
└── events.csv
```

`summary.json` chứa input, method, số frame, FPS, line ratio, direction, count từng class,
tổng count và toàn bộ event. `events.csv` thuận tiện mở bằng Excel hoặc phân tích thống kê.

---

## 17. Kiểm thử

Chạy toàn bộ test:

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
```

Chạy kiểm tra style:

```sh
ruff check .
```

Các nhóm test hiện có:

| File | Phạm vi |
| --- | --- |
| `test_model.py` | Backbone, head, cấu hình và forward model |
| `test_data_and_metrics.py` | YOLO parsing, transform, AP/precision/recall/confusion matrix |
| `test_train.py` | Training smoke test, TensorBoard, validation loss, checkpoint/resume |
| `test_tracker.py` | IoU matching, Kalman/track lifecycle và line crossing |
| `test_pipeline.py` | Video pipeline và artifact đầu ra |

Test pass không chứng minh model đạt accuracy tốt; nó chứng minh logic phần mềm đang hoạt
động theo kỳ vọng. Chất lượng ML vẫn phải được đo trên validation/test đúng miền dữ liệu.

---

## 18. Lỗi thường gặp

### 18.1. `CUDA is not available`

Kiểm tra:

```sh
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi
```

Nếu chỉ muốn smoke test, truyền `--device cpu --no-amp`, nhưng Faster R-CNN train trên CPU
sẽ chậm.

### 18.2. CUDA out of memory

Giảm theo thứ tự:

1. `--batch-size 1`.
2. Tăng `--accumulate` để giữ effective batch.
3. Dùng `resnet34` hoặc `resnet18`.
4. Giảm `--min-size/--max-size` qua package CLI.
5. Đóng tiến trình khác đang dùng GPU.

Không nên chỉ giảm accumulation vì nó không trực tiếp giảm VRAM của một forward; batch
size và resolution mới là yếu tố chính.

### 18.2.1. WSL2: `CUDA driver error: device not ready` / `Failed to create GPU mapping`

Lỗi này xuất hiện giữa lúc train (ví dụ ở batch 63 của epoch 1), sau đó `nvidia-smi` báo
`GPU access blocked by the operating system` và `torch.cuda.is_available()` trả về
`False` cho tới khi restart WSL. Trong `dmesg` có dòng `dxgkio_make_resident: Ioctl failed: -12`
(hết bộ nhớ GPU).

Nguyên nhân là `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: chế độ này cấp phát
VRAM qua API ánh xạ bộ nhớ ảo của CUDA; trên WSL2 với GPU 4GB, khi VRAM đầy nó thất bại
như trên thay vì báo `CUDA out of memory` thông thường. Dự án **không** bật chế độ này.
Nếu shell của bạn có đặt biến đó thì bỏ đi (`unset PYTORCH_CUDA_ALLOC_CONF`). Nếu sau đó
gặp `CUDA out of memory` thật thì làm theo 18.2.

### 18.3. Không tìm thấy dataset

Đảm bảo có `<data>/<split>/images`. Validation mặc định tên `val`, không phải `valid`.
Ảnh và label phải cùng stem, ví dụ `abc.jpg` và `abc.txt`.

### 18.4. Loss NaN hoặc tăng mạnh

- Kiểm tra label nằm trong `[0,1]`, box có width/height dương.
- Giảm learning rate.
- Thử `--no-amp` để xác định lỗi numerical precision.
- Xem `train/grad_norm` và `train/skipped_batches`.
- Kiểm tra class ID chỉ là 0 hoặc 1.

### 18.5. Resume báo config không khớp

Lặp lại đúng `--data`, `--batch-size`, `--accumulate`, `--lr`, backbone, seed và output
của run cũ. Không dùng `best_map50.pt` để resume; dùng `last.pt` vì best checkpoint không
có optimizer/scaler/RNG.

### 18.6. TensorBoard không có dữ liệu

- Kiểm tra đúng đường dẫn `.../logs/tensorboard`.
- Đợi đủ lô scalar đầu tiên (vài chục optimizer step) hoặc hết epoch.
- Refresh trình duyệt.
- Không trỏ TensorBoard trực tiếp vào thư mục `checkpoints`.

### 18.7. FFmpeg/FFprobe không tồn tại

Cài FFmpeg và xác nhận cả hai lệnh có trong PATH:

```sh
ffmpeg -version
ffprobe -version
```

### 18.8. Đếm thiếu hoặc đếm dư

- Chọn đúng `--direction`.
- Đặt line ở vùng object di chuyển ổn định, tránh khu vực che khuất.
- Điều chỉnh `--min-score`; quá cao làm mất detection, quá thấp tăng false positive.
- Tăng processing width nếu trái cây quá nhỏ.
- Kiểm tra detector trước, rồi mới chỉnh tracker; tracker không thể sửa detector quá yếu.

---

## 19. Giới hạn hiện tại và hướng phát triển

### 19.1. Giới hạn

- Model train từ random initialization nên cần nhiều dữ liệu hơn pretrained transfer learning.
- COCO orange/apple khác miền dây chuyền: góc camera, ánh sáng, motion blur và che khuất.
- Metric detector không trực tiếp đo count accuracy của cả video.
- Tracker chỉ dựa trên motion và IoU, chưa có appearance ReID.
- Chưa có calibration confidence, PR curve hoặc COCO mAP@[0.5:0.95].
- Video output không giữ audio.
- Chưa có deployment service, UI hoặc real-time camera input.

### 19.2. Hướng cải thiện ưu tiên

1. Thu thập và annotate video dây chuyền thực, chia theo recording session.
2. Tạo bộ test video có ground-truth count để đo MAE/count accuracy và duplicate rate.
3. Phân tích false positive, false negative, class confusion theo điều kiện ánh sáng.
4. Tune confidence, line position, `min_hits`, `max_missing` trên validation video.
5. Cân nhắc pretrained backbone nếu quy định đồ án cho phép.
6. Bổ sung ReID hoặc appearance feature nếu ID switch là lỗi chính.
7. Export TorchScript/ONNX hoặc dùng inference engine nếu cần real-time.

### 19.3. Chỉ số nên báo cáo trong đồ án

- AP50 từng class và mAP50.
- Precision, recall, confusion matrix.
- Train loss và validation loss theo epoch.
- FPS inference trên đúng phần cứng triển khai.
- Count error trên từng video: `abs(predicted_count - true_count)`.
- Tổng duplicate count, missed crossing và wrong-class crossing.

---

## 20. Thứ tự đọc source code

Để hiểu dự án nhanh mà không bị rối, nên đọc theo thứ tự:

1. `src/fruit_counter/__main__.py`: các lệnh CLI và tham số train/evaluate/count.
2. `src/fruit_counter/model.py`: đọc CNN, ResNet, FPN, RPN, RoI box head.
3. `src/fruit_counter/dataset.py`: hiểu format label, mapping class và augmentation.
4. `src/fruit_counter/train.py`: config, LR schedule, checkpoint và metadata.
5. `src/fruit_counter/engine.py`: forward/backward/optimizer, validation và log.
6. `src/fruit_counter/evaluate.py`: công thức metric và validation loss.
7. `fruit-counter evaluate` (trong `__main__.py`): đánh giá checkpoint độc lập.
8. `src/fruit_counter/fruit_counter_net.py`: adapter từ model sang detection.
9. `src/fruit_counter/tracker.py`: Kalman, Hungarian, voting và line crossing.
10. `src/fruit_counter/pipeline.py`: nối detector → tracker → FFmpeg output.
11. `tests/`: xem hành vi kỳ vọng và các edge case.

Tóm tắt một câu:

```text
Dataset YOLO → Faster R-CNN học box/class → evaluator chọn best model
→ detector chạy từng frame → tracker giữ ID → line counter tạo sự kiện đếm.
```

Tài liệu này mô tả trạng thái code hiện tại. Khi thay đổi class, kiến trúc, threshold,
dataset split hoặc command-line option, cần cập nhật tài liệu cùng commit để tránh lệch
giữa mô tả và implementation.
