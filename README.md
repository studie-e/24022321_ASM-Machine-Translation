# Dịch máy Anh - Việt: Transformer và Seq2Seq Attention

Cài đặt và thực nghiệm hai mô hình dịch máy bằng **PyTorch** trên **IWSLT 2015 English-Vietnamese**: Transformer Encoder-Decoder và Seq2Seq Bi-LSTM với Luong Attention, input feeding. Hai mô hình dùng chung corpus, tokenizer BPE, quy trình đánh giá và thuật toán giải mã.

Đây là bản tái triển khai kiến trúc theo hai nguồn của đề bài (`demo_transformer.ipynb` và TensorFlow NMT), có điều chỉnh cấu hình và pipeline. Không phải chạy nguyên repository TensorFlow hoặc tái lập chính xác mọi siêu tham số của nguồn gốc. Không sử dụng mô hình tiền huấn luyện hay dữ liệu song ngữ bổ sung.

## Kết quả thực nghiệm

Đánh giá trên **1.268 cặp câu của tst2013**, sử dụng checkpoint tốt nhất theo **dev BLEU greedy trên tst2012**.

| Mô hình | Epoch tốt nhất | Test BLEU greedy | Test BLEU beam 5 | chrF2 beam 5 | Tham số |
|---|---:|---:|---:|---:|---:|
| Transformer | 28 | 28.63 | **29.78** | **48.70** | 39,737,344 |
| Seq2Seq + Luong Attention | 10 | 26.83 | 28.18 | 47.24 | 20,005,889 |

- Transformer chạy 30 epoch; Seq2Seq dừng sớm ở epoch 15, với `patience=5`.
- Beam 5 tăng BLEU so với greedy: Transformer **+1.15**, Seq2Seq **+1.36**.
- Tổng thời gian **train thuần**: Transformer **4,282.43 giây (71.37 phút)**; Seq2Seq **3,825.89 giây (63.76 phút)**. Không gồm validation, giải mã test và lưu checkpoint.
- Train trung bình mỗi epoch: Transformer **142.75 giây**, Seq2Seq **255.06 giây**. Tổng thời gian không so như cùng số epoch vì hai run dừng ở thời điểm khác nhau.
- Phần cứng: **Tesla T4**; PyTorch **2.14.1+cu130**; train với AMP, đánh giá FP32.

Số liệu lấy từ output của hai notebook thực nghiệm và `comparison/comparison.json`. Đây là kết quả của một run cho mỗi mô hình; chưa có trung bình nhiều seed hoặc kiểm định ý nghĩa thống kê. Transformer có gần gấp đôi số tham số, nên không quy toàn bộ chênh lệch BLEU cho kiến trúc.

## Cấu trúc project

| Đường dẫn | Nội dung |
|---|---|
| `common/` | Dữ liệu, tokenizer, loss, training engine, checkpoint, giải mã và metric |
| `translate_transformers/` | Cấu hình, mô hình và lệnh train Transformer |
| `translate_seq2seq/` | Cấu hình, mô hình và lệnh train Seq2Seq |
| `notebooks/` | Hai notebook Colab: setup, kiểm thử, train/resume, biểu đồ, đánh giá |
| `scripts/` | Chuẩn bị dữ liệu, sanity check, resume và benchmark |
| `tests/` | Kiểm thử mô hình, dữ liệu, gradient, decoding, AMP và resume |
| `data/iwslt15_en_vi/` | Các file train, tst2012, tst2013 theo hai ngôn ngữ |
| `data/tokenizer/` | Tokenizer SentencePiece BPE và thông tin kiểm tra |
| `evaluate.py` | Đánh giá checkpoint tốt nhất hoặc dịch câu đầu vào |
| `compare.py` | Kiểm tra điều kiện so sánh, xuất bảng và biểu đồ |
| `comparison/` | Kết quả thực nghiệm đã xuất; sao chép thư mục này từ Drive khi công bố |

## Dữ liệu và tiền xử lý

| Tập | File | Số cặp gốc | Số cặp sử dụng |
|---|---|---:|---:|
| Train | `train.en`, `train.vi` | 133,317 | 132,009 |
| Dev | `tst2012.en`, `tst2012.vi` | 1,553 | 1,553 |
| Test | `tst2013.en`, `tst2013.vi` | 1,268 | 1,268 |

- Hai tokenizer SentencePiece **BPE 8.000 token mỗi phía**, chỉ học từ train, normalization `identity`.
- Giữ hoa/thường và văn bản đã tách token theo quy ước Moses; không thêm bước tách từ tiếng Việt.
- Train loại cặp rỗng hoặc có phía dài quá 100 token sau khi thêm token đặc biệt. Dev/test không lọc câu dài.
- Token đặc biệt: PAD=0, UNK=1, BOS=2, EOS=3. Loss bỏ PAD, có tính EOS.
- Batch ghép câu gần độ dài, giới hạn `batch_size × max_length` theo độ dài lớn nhất giữa nguồn và đích; `max_tokens=4096`, không phải batch cố định 4.096 câu.

Corpus gốc có **12 cặp trùng train–dev, 12 cặp train–test, 3 cặp dev–test** theo so sánh chuỗi sau `strip`. Thí nghiệm giữ nguyên split chuẩn, không tuyên bố các split hoàn toàn không trùng. Chi tiết nằm trong `data/tokenizer/data_stats.json`.

## Cấu hình mô hình

| Thành phần | Transformer | Seq2Seq |
|---|---|---|
| Embedding / hidden | 512 / 512 | 512 / 512 |
| Encoder | 6 lớp, Pre-Norm | Bi-LSTM 1 lớp, 512 mỗi chiều |
| Decoder | 6 lớp, Pre-Norm | LSTM 2 lớp, hidden 512 |
| Attention | 4 heads, scaled dot-product | General Luong + hệ số scale học được |
| FFN | 1024, ReLU | Không áp dụng |
| Dropout | 0.3; attention 0.1 | 0.2 |
| Optimizer | AdamW, β=(0.9, 0.98), ε=1e-9, weight decay=1e-4 | Adam, lr=1e-3 |
| Learning rate | Inverse-square-root, warmup=4000, scale=2 | Cố định |
| Gradient clipping | 1.0 | 5.0 |

Cả hai dùng teacher forcing, label smoothing 0.1, seed 42, target embedding/output weight tying, tối đa 30 epoch, early stopping theo dev BLEU với patience 5. Seq2Seq dùng projection từ encoder 1024 xuống 512, bridge trạng thái và input feeding; các phép LSTM giữ FP32 trong khi các phép chiếu phù hợp dùng autocast.

Cấu hình trên đối chiếu với mã nguồn và các lệnh đã chạy trong notebook. Khi tái lập một checkpoint cụ thể, `config.json` trong run là nguồn cấu hình chính thức.

## Cài đặt và kiểm thử

Chạy từ thư mục gốc project, Python 3.10 trở lên:

```bash
python -m pip install -r requirements-dev.txt
python scripts/prepare.py
python -m pytest -q tests
```

Nếu thiếu corpus:

```bash
python scripts/download_data.py
python scripts/prepare.py
```

## Chạy trên Colab

1. Mở notebook tương ứng trong `notebooks/`, chọn GPU.
2. Khi chạy mới, sửa cell cấu hình có sẵn:

```python
MODEL = "transformer"  # Dùng "seq2seq" trong notebook Seq2Seq
MODE = "new"
RUN_NAME = ""
TOTAL_EPOCHS = 30
USE_AMP = True
RUN_FINAL_EVALUATION = False
OTHER_RUN_NAME = ""
```

3. Run all, upload ZIP project và cấp quyền Google Drive. Notebook kiểm thử trước khi train, lưu checkpoint vào `/content/drive/MyDrive/nmt_runs/<RUN_NAME>/`.
4. Xem dev loss/dev BLEU và chốt run. Chỉ sau đó bật `RUN_FINAL_EVALUATION=True`, chạy lại cell cấu hình và bước đánh giá cuối. Không dùng test để chọn epoch hoặc sửa siêu tham số.
5. Nếu đã train xong, không Run all với `MODE="new"`; thao tác này tạo một lượt train mới.

## Train bằng dòng lệnh

```bash
python translate_transformers/train.py --epochs 30 --amp --output-dir runs/transformer
python translate_seq2seq/train.py --epochs 30 --amp --output-dir runs/seq2seq
```

Chạy tuần tự khi dùng chung một GPU. Bỏ `--amp` nếu chạy CPU. Không dùng `--small` hoặc `--smoke` nếu muốn tái lập cấu hình trong bảng kết quả.

### Resume

```bash
python scripts/resume_optimized.py --run-dir runs/transformer --epochs 30 --amp
python scripts/resume_optimized.py --run-dir runs/seq2seq --epochs 30 --amp
```

Script đọc cấu hình từ `last.pt`, khôi phục trọng số, optimizer, scheduler, scaler, RNG và lịch sử. `--epochs` là tổng số epoch mục tiêu, không phải số epoch chạy thêm. Resume yêu cầu mục tiêu lớn hơn epoch đã hoàn thành; trạng thái early stopping vẫn được giữ.

Trong Colab, đặt `MODE="resume"` và điền chính xác tên thư mục run vào `RUN_NAME`. Khi ngắt giữa epoch, resume từ epoch hoàn chỉnh gần nhất. Đổi AMP khi resume được ghi nhận nhưng không bảo đảm tiếp tục bit-for-bit như run không đổi precision.

## Đánh giá và so sánh

```bash
python evaluate.py --run-dir runs/transformer --data-dir data/iwslt15_en_vi --beam 1 5 --show 8
python evaluate.py --run-dir runs/seq2seq --data-dir data/iwslt15_en_vi --beam 1 5 --show 8
python compare.py --runs runs/transformer runs/seq2seq --output-dir comparison --plots
```

Đổi đường dẫn `runs/...` theo nơi lưu checkpoint. `evaluate.py` dùng **best.pt**, không mặc định dùng epoch cuối. Kết quả nằm trong `<run>/evaluation/test/`; `compare.py` xuất `comparison.csv`, `comparison.json`, `test_bleu.png`, `training_curves.png` và kiểm tra tính tương thích dữ liệu, tokenizer, metric, decoding.

Hai run của kết quả đã báo cáo:

- `transformer_20261110_174642`: 30 epoch, best epoch 28, dev BLEU 26.07.
- `seq2seq_20261110_194635`: 15 epoch, best epoch 10, dev BLEU 24.16.

### Quy ước chấm điểm

- SacreBLEU **2.6.0**; BLEU thang 0–100, `tokenize="none"`, phân biệt hoa/thường, smoothing `exp`.
- BLEU signature: `nrefs:1|case:mixed|eff:no|tok:none|smooth:exp|version:2.6.0`.
- chrF2 đo trên cùng văn bản Moses-tokenized, không phải kết quả detokenized.
- Giải mã FP32; greedy và beam size 5; tối đa 160 token mới; length penalty `((5 + L) / 6)^1.0`.
- `train_nll` và `dev_nll` không smoothing; hàm loss tối ưu có smoothing. Không so trực tiếp loss khi tokenizer khác nhau.
- Seq2Seq có một câu chạm giới hạn sinh ở mỗi chế độ; Transformer không có. Giữ nguyên các câu đó trong phép đánh giá.

Dịch câu đơn:

```bash
python evaluate.py --run-dir runs/transformer --text "I love machine translation ." --beam 1 5
```

## Ghi chú kỹ thuật về cài đặt

### Transformer

- **Pre-Norm thay vì Post-Norm**: LayerNorm được đặt *trước* sublayer (`norm → sublayer → residual`), không phải sau như bài báo gốc "Attention Is All You Need". Pre-Norm thường ổn định hơn khi huấn luyện sâu và thường được ưa dùng trong các cài đặt hiện đại.
- **Positional Encoding động**: `PositionalEncoding` tự mở rộng bảng sin/cos nếu chuỗi đầu vào dài hơn `initial_length`. Điểm thú vị là embedding được nhân với `√d_model` *trước* khi cộng PE, đúng theo paper gốc.
- **Xavier init + init embedding chuẩn hóa**: Ma trận trọng số `dim > 1` dùng Xavier uniform, còn embedding dùng normal với std = `d_model⁻⁰·⁵` — kết hợp giúp cân bằng scale giữa hai loại tham số.
- **Tie embeddings**: `output.weight = tgt_embedding.weight` — chia sẻ trọng số giữa embedding và lớp đầu ra, giảm tham số và thường cải thiện generalization trên vocab nhỏ-trung bình.
- **`F.scaled_dot_product_attention`**: Dùng kernel fused của PyTorch (FlashAttention-style nếu hỗ trợ), tốt hơn self-implementation thuần tuý về tốc độ và bộ nhớ.

### Seq2Seq

- **Bi-LSTM encoder → LSTM decoder**: Encoder 1 lớp Bi-LSTM (1024 chiều tổng), decoder 2 lớp LSTM đơn hướng (512). Trạng thái ẩn của encoder được chiếu xuống 512 (`bridge_h`, `bridge_c` qua `tanh`) để khởi tạo trạng thái decoder.
- **Orthogonal init cho recurrent weights**: `weight_hh` được khởi tạo orthogonal cho từng gate riêng — kỹ thuật được biết là giúp gradient chảy tốt hơn trong LSTM, giảm vanishing/exploding gradient khi huấn luyện.
- **Forget gate bias = 1**: `bias_ih` phần forget gate được khởi tạo = 1 (`parameter.chunk(4)[1].fill_(1)`). Đây là best practice kinh điển (Jozefowicz et al., 2015) giúp LSTM ghi nhớ thông tin tốt hơn ở giai đoạn đầu huấn luyện.
- **Luong Attention với scale học được**: `scale = nn.Parameter(torch.ones(()))` — hệ số tỉ lệ cho điểm attention là tham số học, không cố định. Khác với scaled dot-product chuẩn dùng `1/√d`.
- **Input feeding**: Context vector từ bước trước (`feed`) được nối với embedding hiện tại trước khi đưa vào LSTM — giúp decoder "nhớ" đã chú ý vào đâu ở bước trước. Tensor `feed` được lưu trong `state`.
- **FP32 cho LSTM, autocast cho projection**: LSTM tính ở FP32 để ổn định số học (LSTM nhạy với precision thấp), trong khi các phép chiếu dùng autocast AMP — đây là cách xử lý AMP thực tế cho RNN thay vì tắt AMP hoàn toàn.

## Phân tích và Nhận xét

### Hiệu năng và quy mô mô hình

Trong thiết lập thực nghiệm từ đầu trên IWSLT 2015 En→Vi, Transformer đạt 29,78 BLEU với beam search kích thước 5, cao hơn 1,60 điểm so với Seq2Seq + Luong Attention (28,18 BLEU). Tuy nhiên, Transformer có khoảng 39,7 triệu tham số, gần gấp đôi Seq2Seq với 20,0 triệu tham số. Kết quả cho thấy sự đánh đổi giữa chất lượng dịch và quy mô mô hình: Seq2Seq sử dụng ít hơn khoảng 49,6% tham số nhưng vẫn đạt kết quả tương đối gần Transformer. Số tham số chưa đủ để kết luận về bộ nhớ hay độ trễ suy luận; các tiêu chí này cần được đo riêng.

### Diễn biến huấn luyện

Seq2Seq đạt dev BLEU tốt nhất tại epoch 10 và dừng tại epoch 15 theo cơ chế early stopping với patience bằng 5. Transformer đạt dev BLEU tốt nhất tại epoch 28 trong tổng số 30 epoch. Như vậy, Seq2Seq đạt checkpoint được chọn sớm hơn theo số epoch trong thí nghiệm này. Tuy nhiên, sự khác biệt còn chịu ảnh hưởng của optimizer, lịch learning rate và cấu hình mô hình, nên chưa thể quy hoàn toàn cho kiến trúc LSTM hay Transformer.

### Ảnh hưởng của beam search

Khi chuyển từ greedy decoding sang beam search kích thước 5, BLEU của Transformer tăng từ 28,63 lên 29,78, còn Seq2Seq tăng từ 26,83 lên 28,18. Mức tăng tương ứng khoảng 1,15 và 1,36 BLEU cho thấy cả hai mô hình đều hưởng lợi từ việc mở rộng không gian tìm kiếm khi giải mã. Seq2Seq có mức cải thiện lớn hơn trong thiết lập này, nhưng kết quả chưa đủ để xác định nguyên nhân ở từng thành phần kiến trúc.

### Thời gian huấn luyện

Thời gian train trung bình mỗi epoch của Transformer khoảng 143 giây, thấp hơn Seq2Seq với khoảng 255 giây, tương ứng Seq2Seq chậm hơn khoảng 1,79 lần. Các số liệu này không bao gồm đánh giá trên dev và lưu checkpoint. Khả năng xử lý song song theo vị trí của Transformer là một yếu tố có thể góp phần tạo ra khác biệt; ngoài ra, cách triển khai và chế độ tính toán cũng ảnh hưởng đến tốc độ. Tổng thời gian train thực tế của Transformer vẫn cao hơn do mô hình được huấn luyện nhiều epoch hơn.

### Giới hạn của kết quả

- Mỗi mô hình được chạy với một seed cố định; chưa đánh giá được độ biến thiên giữa các lần huấn luyện hoặc kết luận về ý nghĩa thống kê của chênh lệch BLEU.
- Dữ liệu có 12 cặp câu trùng giữa train và test theo phép so khớp chuỗi sau `strip`; ảnh hưởng của phần trùng này chưa được định lượng.
- Kết quả chrF2 có cùng thứ hạng với BLEU, cung cấp thêm một góc nhìn nhất quán, nhưng chưa thay thế được đánh giá thủ công chất lượng bản dịch.

### Hướng phát triển

Các thí nghiệm tiếp theo có thể khảo sát kích thước vocabulary, tokenizer dùng chung giữa hai ngôn ngữ và lịch learning rate cho Seq2Seq. Với BPE, tăng vocabulary không mặc nhiên cải thiện chất lượng: cần đánh giá sự đánh đổi giữa độ dài chuỗi, số tham số và khả năng học các token ít xuất hiện. Việc bổ sung dữ liệu cần đi kèm lọc chất lượng, kiểm tra trùng lặp và xem xét sự phù hợp miền dữ liệu. Mọi lựa chọn cấu hình nên dựa trên tập dev; tập test được giữ lại cho đánh giá cuối cùng.

