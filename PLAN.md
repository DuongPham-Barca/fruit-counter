# Kế hoạch đếm trái cây từ video

## Yêu cầu đã chốt

- Đầu vào là **tệp video quay sẵn của dây chuyền trái cây**, xử lý sau khi quay. Hiện chưa có video mẫu.
- Nhận diện và đếm **cam** và **táo** trong video; xuất số cam, số táo, tổng cộng cùng video có đánh dấu kết quả. Đây là hai loại có dây chuyền phân loại thực tế và là phạm vi dự kiến cho bản đầu.
- Mô hình học máy, nếu dùng, phải **huấn luyện từ khởi tạo ngẫu nhiên**. Không tải hoặc dùng trọng số pretrained cho mô hình nhận diện, backbone hay bộ trích xuất đặc trưng.

## Trạng thái hiện tại

- **Mô hình nhận diện: Faster R-CNN** (Ren et al. 2015) gồm backbone ResNet-50, FPN (Lin et al. 2017), RPN, RoIAlign 7×7, box head 4conv-1fc và bộ phân loại 3 lớp (nền/cam/táo). Toàn bộ CNN, GroupNorm và ReLU được khai báo tường minh ở `src/fruit_counter/model.py`.
  - Mọi trọng số khởi tạo ngẫu nhiên. Mọi lớp chuẩn hóa là GroupNorm, theo He et al. 2018, "Rethinking ImageNet Pre-training".
  - Có test chặn tải trọng số và kiểm tra không còn BatchNorm hay lớp bị đóng băng.
  - Anchor có 5 mức (16–256 px), tỉ lệ khung 0.75/1/1.33.
- **Huấn luyện** (entrypoint dễ đọc `train_model.py`, config/checkpoint ở `src/fruit_counter/train.py`, training loop ở `src/fruit_counter/engine.py`):
  - SGD, warmup tuyến tính rồi cosine, AMP, tích lũy gradient (batch 2 × 4).
  - Cạnh ngắn của ảnh chọn ngẫu nhiên trong 416–512 px.
  - Mỗi epoch đánh giá mAP@0.5 theo lớp trên tập validation, lưu `checkpoints/best_map50.pt` và `checkpoints/last.pt`, có thể chạy tiếp (resume).
- **Theo dõi và đếm** (`tracker.py`):
  - Kiểu ByteTrack: lọc Kalman vận tốc không đổi, ghép bằng thuật toán Hungarian theo IoU (không phụ thuộc nhãn), ghép hai vòng với detection điểm cao rồi điểm thấp.
  - Nhãn của track chốt bằng bỏ phiếu có trọng số theo độ tin cậy.
  - Đếm một lần khi tâm track đi từ phía xuất hiện sang phía bên kia đường đếm.
- **Dữ liệu ban đầu:** COCO 2017, chỉ lấy ảnh có cam hoặc táo, bỏ ảnh có khung crowd (`fruit-counter prepare-coco`).
- **Phần cứng:** RTX 3050 Laptop 4GB. ResNet-50 ở 512 px, batch 2 dùng khoảng 3.5GB VRAM, khoảng 0.46 s/batch; suy luận khoảng 9 FPS. Ở 640 px mô hình vượt 4GB và chậm khoảng 10 lần.
- **Bộ nhận diện màu** chỉ còn dùng để kiểm thử pipeline trên clip giả lập (`--detector color-demo`).

## Quy tắc đếm trên dây chuyền

Đặt camera cố định nhìn thấy một đoạn băng chuyền và một **đường đếm**. Theo dõi từng quả bằng ID qua các khung hình; chỉ cộng một lần khi tâm của ID đó đi qua đường theo chiều chuyển động đã chọn. Ghi thời điểm, ID và nhãn `orange` hoặc `apple` cho từng lần cộng. Kiểm tra riêng quả đi sát nhau, bị che khuất ngắn, dừng trên đường, đi ngược chiều hoặc ra rồi vào lại khung hình. Vị trí đường đếm và chiều chuyển động sẽ được cấu hình theo video thật.

## Lộ trình

| Mốc | Công việc | Kết quả |
| --- | --- | --- |
| 1. Khung dự án — đã dựng | Tạo `src/fruit_counter/`, `tests/`, `data/README.md`; dùng `fruit-counter demo --output runs/demo.mp4` và `fruit-counter count --detector color-demo --input runs/demo.mp4 --output runs/demo-count` để thử luồng. | Đọc video, xử lý từng khung hình, ghi video/JSON/CSV và kiểm thử quy tắc cộng một lần bằng clip giả lập. |
| 2. Thu dữ liệu thật | Quay đoạn băng chuyền trống và các video chỉ có cam, chỉ có táo, có cả hai loại, với nhiều tốc độ, ánh sáng và mức che khuất. Gán khung bao + loại ở các khung hình chọn lọc; gán ID và sự kiện qua đường để kiểm tra số đếm. Chia **theo video/lần quay**, không chia ngẫu nhiên từng khung hình. | Bộ train/validation/test và bảng số cam, số táo đúng cho mỗi video. |
| 3. Thuật toán ban đầu | Với camera cố định và nền đơn giản, thử tách vật thể bằng màu/chuyển động; nối vị trí giữa các khung hình thành ID và đếm khi ID qua đường đúng chiều. Dùng kết quả này làm mốc so sánh, không coi là độ chính xác trên cảnh chưa được kiểm tra. | Video minh họa, sự kiện đếm có thời điểm/ID và bảng lỗi. |
| 4. Huấn luyện từ đầu — đã dựng mã, chưa huấn luyện | Faster R-CNN ResNet-50-FPN với GroupNorm; train trước trên COCO cam/táo, sau đó fine-tune trên khung hình băng chuyền đã gán nhãn. Khởi tạo mọi trọng số ngẫu nhiên, lưu seed/cấu hình, huấn luyện trên tập train và chọn cấu hình bằng validation. Kiểm tra cả trọng số của backbone: với một số hàm TorchVision, `weights=None` vẫn để mặc định backbone pretrained, nên phải đặt thêm `weights_backbone=None`. | Checkpoint tự huấn luyện, log huấn luyện và báo cáo trên tập test giữ riêng. |
| 5. Tích hợp và hoàn thiện — đã nối mã (`count --weights`), chờ mô hình đã train | Thay bộ nhận diện ban đầu bằng mô hình đã huấn luyện; xử lý mất dấu, che khuất, ra/vào lại khung hình; xuất video có ID, đường đếm, tổng theo loại và JSON/CSV sự kiện. | Lệnh chạy lặp lại được từ video đầu vào và hướng dẫn cài đặt/chạy trong README. |

## Đánh giá và điều kiện hoàn thành

So sánh **số quả duy nhất theo từng loại trên mỗi video** với số đếm thủ công. Báo cáo số đếm thừa/thiếu, độ chính xác nhận diện theo loại, trường hợp đếm trùng và tốc độ xử lý trên máy mục tiêu. Chốt ngưỡng đạt yêu cầu sau khi có video thật và quy tắc đếm. Clip giả lập chỉ xác nhận luồng xử lý và logic đếm, không chứng minh mô hình nhận diện trái cây chính xác.

## Thông tin còn cần chốt

1. Camera sẽ đặt cố định phía trên hay bên cạnh dây chuyền? Quả chạy theo một chiều hay có thể đi ngược/dừng lại?
2. Video dự kiến dài bao lâu, độ phân giải và FPS khoảng bao nhiêu? (GPU đã xác định: RTX 3050 Laptop 4GB.)

## Tài liệu kỹ thuật

- [OpenCV: đọc video từ tệp](https://docs.opencv.org/4.7.0/dd/d43/tutorial_py_video_display.html).
- [OpenCV: tách nền cho camera cố định](https://docs.opencv.org/4.13.0/d1/dc5/tutorial_background_subtraction.html).
- [Ren et al. 2015, Faster R-CNN](https://arxiv.org/abs/1506.01497); [Lin et al. 2017, FPN](https://arxiv.org/abs/1612.03144); [He et al. 2018, Rethinking ImageNet Pre-training](https://arxiv.org/abs/1811.08883); [Zhang et al. 2022, ByteTrack](https://arxiv.org/abs/2110.06864).
- [TorchVision: tham số trọng số của mô hình nhận diện và backbone](https://docs.pytorch.org/vision/master/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html).
- [CVAT: gán nhãn ảnh và video](https://docs.cvat.ai/docs/getting_started/overview/).
- [TOMRA: ví dụ dây chuyền phân loại táo và trái cây họ cam quýt](https://www.tomra.com/en-gb/food/machines/tomra-5s-advanced).
