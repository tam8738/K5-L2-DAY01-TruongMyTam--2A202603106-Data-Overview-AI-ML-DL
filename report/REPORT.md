# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU (T4)

**Python / PyTorch / Ultralytics:** Python 3 / PyTorch 2.x / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

**1. Chép record hạng 1:** `class_id` = 468 ; `class_name` = "cab" ; `rank` = 1 ; `score` = 0.510915 ; `taxonomy_name` = "ImageNet-1K"

**2. Record này nói về cả bức ảnh hay về một vật thể riêng trong ảnh? Vì sao bạn nghĩ vậy?**

Record này nói về **cả bức ảnh** (toàn cảnh), chứ không phải về một vật thể riêng lẻ.

Lý do:
* **Tên tác vụ (`task`)**: `image_classification` (phân loại ảnh) là tác vụ gán nhãn cho toàn bộ nội dung bức ảnh, khác với `object_detection` (nhận diện vật thể) vốn đi kèm với tọa độ khung bao (bounding box) cho từng đối tượng riêng biệt.
* **Tên phân loại (`class_name`)**: Các nhãn như `cab` (xe taxi), `minibus` (xe buýt nhỏ), `restaurant` (nhà hàng), hay `dining_table` (bàn ăn) phản ánh ngữ cảnh tổng thể hoặc đối tượng chính bao trùm toàn bộ bố cục ảnh theo danh mục của ImageNet-1K.
* **Thông tin kích thước (`image_width`, `image_height`)**: Dữ liệu lưu kích thước toàn cục của bức ảnh ($640 \times 428$ hoặc $480 \times 640$) thay vì tọa độ vùng cắt (crop) hay tọa độ $[x, y, w, h]$ của một vật thể cụ thể.

**3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?**

Danh sách tên lớp (`class list`) **đi kèm theo taxonomy (tập dữ liệu chuẩn) và được lưu sẵn trong checkpoint**, chứ model không tự nghĩ ra.

Cụ thể:
* **Thuộc về Taxonomy:** Các model được huấn luyện trên tập dữ liệu chuẩn (như ImageNet-1K gồm 1000 danh mục cố định đã được gán nhãn từ trước).
* **Lưu trong Checkpoint (`.pt`):** Khi huấn luyện xong, trọng số và metadata (ánh xạ từ `class_id` sang `class_name`) được đóng gói chung vào file checkpoint để hiển thị tên lớp dễ đọc thay vì chỉ trả về số ID.

**4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?**

Việc ghi đồng thời cả `class_id`, `class_name` và `taxonomy_name` thay vì chỉ dùng mỗi tên lớp là cần thiết để đảm bảo tính tối ưu cho máy tính, tính trực quan cho con người và tính chính xác về mặt ngữ cảnh.

Cụ thể:
* **`class_id` (Mã số định danh):** Giúp máy tính xử lý, so sánh và tính toán dữ liệu số nhanh chóng, chính xác, đồng thời tránh các lỗi liên quan đến chuỗi ký tự hay ngôn ngữ.
* **`class_name` (Tên lớp trực quan):** Giúp con người đọc hiểu ngay nội dung (ví dụ: biết là "cab" thay vì phải tra cứu mã số 468) khi xem báo cáo, log hoặc giao diện.
* **`taxonomy_name` (Tên hệ thống phân loại):** Xác định rõ nguồn gốc bộ dữ liệu chuẩn (như ImageNet-1K hay COCO-80), tránh nhầm lẫn ý nghĩa khi các tập dữ liệu khác nhau có thể dùng chung ID hoặc tên gọi.

**5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn một lớp?**

Nếu trong ảnh có nhiều vật thể hoặc nhiều chủ đề khác nhau, hướng dẫn gán nhãn (`guideline`) cần phải quy định rõ ràng **tiêu chí ưu tiên** để người gán nhãn chọn ra **một** lớp duy nhất. 

Cụ thể, guideline cần làm rõ các điểm sau:
* **Đối tượng chính (Primary Subject):** Quy định vật thể nào chiếm diện tích lớn nhất, nằm ở vị trí trung tâm, hoặc là điểm nhấn cốt lõi thu hút ánh nhìn đầu tiên của bức ảnh thì được ưu tiên chọn làm nhãn.
* **Mức độ bao quát (Global vs. Local Context):** Xác định rõ trong trường hợp ảnh chụp một khung cảnh tổng thể (như phòng khách, đường phố) chứa nhiều vật dụng nhỏ bên trong, người gán nhãn phải chọn nhãn mô tả **bối cảnh toàn cảnh** hay chọn nhãn của **vật thể nổi bật nhất**.
* **Thứ tự ưu tiên theo danh mục (Hierarchy/Priority Rules):** Trong các tập dữ liệu đa chủ đề, nếu xuất hiện các lớp thuộc nhóm quan trọng cần thu thập (ví dụ: ưu tiên các lớp phương tiện giao thông trước rồi mới đến phông nền), guideline cần liệt kê rõ thứ tự ưu tiên này.
* **Quy tắc đồng thuận khi có độ phân tán (Ambiguity Handling):** Hướng dẫn cách xử lý khi có hai đối tượng có trọng số tương đương nhau (ví dụ: quy định gán nhãn theo vật thể ở gần ống kính hơn hoặc vật thể tương tác trực tiếp với hành động chính).

**6. Vì sao không được lấy `score` của model làm ground truth?**

Không được lấy `score` (độ tin cậy) của model làm ground truth (nhãn chuẩn xác thực tế) vì những lý do sau:
* **`Score` chỉ thể hiện mức độ tự tin của model, không phải sự thật khách quan:** `Score` là xác suất do chính mô hình tính toán dựa trên phân phối học được từ dữ liệu huấn luyện, chứ không phản ánh chính xác $100\%$ bản chất thực tế của hình ảnh.
* **Model hoàn toàn có thể sai lầm (Hiện tượng quá tự tin - Overconfidence):** Một mô hình có thể đưa ra `score` rất cao cho một dự đoán hoàn toàn sai lệch do gặp ảnh nhiễu, góc chụp lạ hoặc điểm mù của kiến trúc mạng (ví dụ trên ảnh `kitchen`, model phân loại nhầm thành `gong` - cái chiêng với score đứng hạng 1 là $0.420191$ do nhìn nhầm các đáy chảo đồng treo trên tường).
* **Vi phạm tính độc lập trong đánh giá:** Ground truth đòi hỏi phải là tiêu chuẩn vàng khách quan do con người kiểm duyệt theo quy tắc. Nếu lấy `score` của chính model làm chuẩn, hệ thống sẽ rơi vào vòng lặp tự quy chiếu (tự quy định mình đúng), làm mất hoàn toàn ý nghĩa của việc kiểm định và đo lường chất lượng thực tế.

---

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

**1. Chép một record:**
* `class_name` = "bowl" ; `score` = 0.719063 ; `bbox_xyxy` = [32.65, 342.12, 100.16, 384.93] ; `bbox_width` = 67.51 ; `bbox_height` = 42.81

**2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?**
* **Góc trên-trái ($x_{\min}, y_{\min}$):** Nằm ở tọa độ $x = 32.65\text{ pixel}$ (cách mép trái ảnh) và $y = 342.12\text{ pixel}$ (cách mép trên ảnh).
* **Góc dưới-phải ($x_{\max}, y_{\max}$):** Nằm ở tọa độ $x = 100.16\text{ pixel}$ (cách mép trái ảnh) và $y = 384.93\text{ pixel}$ (cách mép trên ảnh).
* **Kích thước:** Khung bao (bounding box) này có chiều rộng là $67.51\text{ pixel}$ và chiều cao là $42.81\text{ pixel}$.

**3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?**

Dựa trên kết quả thực nghiệm của ảnh `kitchen`:
* **Ngưỡng `0.60`:** Có **6 prediction** (chỉ giữ lại các hộp có độ tin cậy $\ge 0.60$ gồm: `person` $0.91$, `bowl` $0.72$, `bowl` $0.70$, `oven` $0.69$, `oven` $0.63$, `person` $0.61$).
* **Ngưỡng `0.35`:** Có **11 prediction** (bao gồm 6 hộp ở trên và lấy thêm 5 hộp có độ tin cậy trong khoảng từ $0.35$ đến dưới $0.60$ gồm: `bowl` $0.50$, `bowl` $0.46$, `cup` $0.45$, `cup` $0.38$, `bowl` $0.38$).

**4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người reviewer phải làm thêm việc gì?**

Khi hạ ngưỡng (giữ lại nhiều dự đoán hơn), người reviewer phải **kiểm duyệt thủ công nhiều hộp nhiễu hơn (False Positives)**. Việc này tiêu tốn thêm thời gian để xóa bỏ các hộp nhận diện sai (nhận nhầm bối cảnh thành vật thể hoặc vẽ trùng lặp), tinh chỉnh lại các bounding box vẽ quá rộng/quá hẹp, và sửa lại các nhãn bị phân loại sai do model dự đoán ở mức độ tự tin thấp.

**5. Viết một quy tắc box để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).**
* **Quy tắc Bám sát (Tight Fit):** Bounding box phải bao bọc trọn vẹn toàn bộ vùng diện tích có thể nhìn thấy được của đối tượng, sao cho 4 cạnh của hộp chạm sát vào các điểm ngoài cùng (trên, dưới, trái, phải) của đối tượng đó. Tuyệt đối không cắt lẹm vào phần thân vật thể và không để thừa khoảng trống nền (background) bên trong hộp quá $1 - 2\text{ pixel}$.

**6. Trên ảnh `kitchen`, chỉ ra một vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?**
* **Vật thể trên ảnh `kitchen`:** Chiếc lò nướng (`oven` với score $0.686791$) ở rìa trái bức ảnh bị cắt mất một phần bởi mép khung hình (tọa độ $x_{\min} = 0.11$ nằm sát lề trái). Tương tự, phần bàn tay/cánh tay người (`person` với score $0.610931$, $x_{\min} = 0.12$) ở góc trái dưới chỉ lộ ra một phần nhỏ từ mép ảnh, hoặc chiếc khay/bát (`bowl` $0.381215$) bị che khuất một phần bởi khuôn bánh phía trước.
* **Quyết định:** Viết thành **quy tắc thẳng vào Guideline**, không cần escalate lên Lab Coach.
* **Vì sao:** Cắt mép (Truncation) và che khuất (Occlusion) là những tình huống xuất hiện thường xuyên trong dữ liệu thực tế của bài toán Object Detection. Nếu mỗi lần gặp vật bị che đều phải escalate hỏi Lab Coach thì quy trình gán nhãn sẽ bị đình trệ. Do đó, cần định nghĩa quy tắc chuẩn ngay trong Guideline để mọi Annotator áp dụng đồng nhất (Ví dụ: *"Chỉ vẽ bounding box bao quanh phần diện tích thực tế nhìn thấy được của vật thể, không tự suy đoán để vẽ tràn ra ngoài biên ảnh hoặc lấn sang vùng bị vật khác che khuất; nếu vật bị che khuất quá $80\%$ không rõ hình dạng thì bỏ qua không gán nhãn"*).

---

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

**1. Chép một record:**
* `instance_id` = "kitchen-001"
* `class_name` = "person"
* `score` = 0.899318
* Số điểm polygon (`polygon_point_count`) = 348
* 5 điểm đầu của `polygon_xy` = `[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`

**2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?**
* Nếu Bounding Box (hộp bao hình chữ nhật) chỉ cung cấp thông tin về vị trí và kích thước bao quát cực đại (chiều rộng/chiều cao lớn nhất) của vật thể, thì **Polygon (đa giác)** cung cấp thông tin **chính xác về đường viền (contour) và hình dáng thực tế** của vật thể ở mức độ điểm ảnh. Nó giúp tách bạch rõ ràng các pixel thực sự thuộc về vật thể (foreground) khỏi các pixel thuộc về môi trường xung quanh (background) hoặc các vật thể khác nằm lọt thỏm trong cùng một khu vực hộp bao.

**3. `instance_id` dùng để làm gì? Nó không phải class ID và không phải tracking ID — bạn hiểu câu này thế nào?**
* **`instance_id`** dùng để **định danh duy nhất một cá thể (một đối tượng độc lập)** xuất hiện trong phạm vi một bức ảnh tĩnh (ví dụ: `"kitchen-001"`, `"kitchen-002"`).
* **Không phải `class_id`:** Vì trong một ảnh có thể có nhiều đối tượng cùng thuộc một lớp (ví dụ: ảnh `kitchen` có tới 4 đối tượng cùng mang `class_id = 45` là `"bowl"`, nhưng chúng là 4 chiếc bát rời rạc nên phải có 4 `instance_id` khác nhau là `kitchen-002`, `kitchen-003`, `kitchen-006`, `kitchen-010` để phân biệt từng chiếc).
* **Không phải `tracking_id`:** Vì `instance_id` chỉ có giá trị phân biệt các cá thể nội trong **một bức ảnh độc lập**, hoàn toàn không có chức năng liên kết hay theo dõi (track) xem đối tượng `kitchen-001` ở khung hình (frame) hiện tại có phải là cùng một người ở khung hình tiếp theo trong chuỗi video hay không.

**4. Viết một quy tắc vẽ biên mask để người gán nhãn làm theo.**
* **Quy tắc Bám sát viền (Tight Contour / Pixel-accurate Boundary):** Đường đa giác (polygon) phải đi dọc theo đúng ranh giới hình học bên ngoài của vật thể. Không được chấm điểm quá thưa làm đường viền cắt góc thẳng qua các chi tiết cong của vật thể, không vẽ lẹm ra ngoài lấy thêm pixel của phần nền (background), và không vẽ hụt vào trong làm khuyết thiếu chi tiết của đối tượng.

**5. Trên ảnh `kitchen`, chỉ ra một chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?**
* **Vị trí trên ảnh:** Trên mặt bàn gỗ (`dining table` - `kitchen-005`), các khuôn bánh/bát (`bowl` - `kitchen-002`, `kitchen-003`, `kitchen-006`) đặt chạm sát nhau và nằm đè trực tiếp lên mặt bàn khiến mask của mặt bàn bị khoét lỗ hoặc chồng lấn ranh giới với các bát. Ngoài ra, phần viền cơ thể của người đầu bếp (`person` - `kitchen-001`) cũng che khuất phần bệ bếp phía sau.
* **Quyết định:** Sử dụng **Guideline**.
* **Vì sao:** Hiện tượng hai vật chạm nhau (Touching objects) hoặc vật trên đè lên vật dưới (Occlusion) là tính chất tự nhiên xuất hiện ở hầu hết các bức ảnh đời thực. Cần quy định rõ nguyên tắc xử lý ngay trong Guideline để người gán nhãn thực hiện nhất quán mà không phải dừng lại hỏi Lab Coach liên tục (Ví dụ: *"Đối với hai vật chạm nhau hoặc đặt đè lên nhau, đường biên mask của hai đối tượng phải chia sẻ ranh giới chung khớp khít nhau, không để hở khoảng trống ở giữa và không vẽ mask chồng đè lên vùng diện tích nhìn thấy của vật nằm phía trước"*).

---

## 4. Vòng đời và kiểm tra chất lượng

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| **Phân loại ảnh** | 1 nhãn lớp (`class_id` / `class_name`) đại diện cho toàn bộ bức ảnh. | Ảnh `kitchen` có nhiều đồ vật làm model đoán nhầm hạng 1 là `gong` (cái chiêng - score $0.42$) do nhìn nhầm đáy chảo đồng; ảnh `traffic` có cả xe buýt và ô tô con nhưng chỉ được chọn 1 nhãn (`cab`). | Xem xét bố cục toàn cảnh và tuân theo quy tắc ưu tiên trong Guideline để gán 1 nhãn đúng nhất cho cả bức ảnh. | Kiểm tra xem nhãn được gán có đúng với chủ đề chính của toàn ảnh và tuân thủ đúng quy tắc ưu tiên khi ảnh có nhiều vật thể hay không. |
| **Phát hiện vật thể** | Tập hợp các khung hình chữ nhật (`bbox_xyxy` kèm `class_id`) bao quanh từng vật thể mục tiêu. | Trên ảnh `kitchen`, lò nướng (`oven` $0.69$) và cánh tay người (`person` $0.61$) ở rìa trái bị cắt mép ảnh ($x_{\min} \approx 0.1$); các chiếc cốc (`cup` $0.45$, $0.38$) nằm sát nhau và bị che một phần. | Vẽ bounding box bám sát (tight fit) phần diện tích nhìn thấy của từng vật thể, gán đúng nhãn lớp và bỏ qua các dự đoán nhiễu (false positives). | Kiểm tra xem có bỏ sót vật thể (false negatives) không, nhãn lớp có đúng không, và 4 cạnh của box đã ôm sát phần nhìn thấy của vật bị che/cắt mép chưa. |
| **Instance segmentation** | Đa giác (`polygon_xy`) hoặc mặt nạ pixel (mask) bao quanh từng cá thể độc lập (`instance_id`) kèm `class_id`. | Mask của mặt bàn (`dining table` `kitchen-005` với $598$ điểm) bị các vật đặt trên bàn (`bowl`, khay bột) che khuất làm đường biên bị chia cắt phức tạp; bó lá góc trái bị nhận nhầm là `potted plant` ($0.63$). | Xóa các mask nhận diện sai (như `potted plant`), vẽ hoặc chỉnh sửa các điểm đa giác (`polygon_xy`) bám sát viền từng cá thể, tách bạch ranh giới nơi hai vật chạm nhau. | Kiểm tra độ chính xác của đường viền đa giác (có bị lẹm vào vật hay dư ra nền không), kiểm tra sự tách biệt giữa các `instance_id` cùng lớp đứng cạnh nhau. |

---

## 5. An toàn dữ liệu

**1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì?**
Quy tắc bảo vệ dữ liệu cốt lõi cho bài lab này là **đảm bảo ẩn danh hóa (anonymization) và bảo mật thông tin cá nhân (PII)**: tuyệt đối không đưa họ tên, MSSV, email, số điện thoại hoặc bất kỳ dữ liệu nhạy cảm nào vào nội dung file `REPORT.md`, các file JSON/PNG output, hay lịch sử commit code (họ tên và MSSV chỉ được đặt ở tên của GitHub repository theo đúng quy định). Đồng thời, tuân thủ việc ghi nguồn tác giả và giấy phép bản quyền ảnh gốc trong file `IMAGE_ATTRIBUTION.md`.

**2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?**
Nếu phát hiện hình ảnh hoặc dữ liệu chứa thông tin nhạy cảm, vi phạm quyền riêng tư hoặc nằm ngoài phạm vi cho phép của bài thực hành, tôi sẽ **lập tức dừng thao tác** (không chia sẻ, không phát tán hay tự ý xử lý tiếp) và báo cáo ngay (escalate) cho **Lab Coach** hoặc giảng viên phụ trách môn học để nhận hướng dẫn xử lý.

---

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/` (Đã hoàn thành đánh dấu `[x]`):

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS