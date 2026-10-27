# Format âm vị G2P / XPhoneBERT (vie-n)

Khảo sát ngày 2026-10-27. `text2phonemesequence==0.1.4`, model G2P `charsiu/g2p_multilingual_byT5_small_100`,
tokenizer `vinai/xphonebert-base`. Chạy thật trên Kaggle (kernel private `triuinhhi/xphonebert-g2p-probe`, CPU),
vì Windows Application Control chặn DLL của torch trên máy local.

File đi kèm:
- `vie-n.tsv`: từ điển vie-n (bản copy, giống hệt file mà thư viện tải về).
- `baseline_g2p_probe.json`: output G2P cho các câu probe và toàn bộ 226 câu `metadata.csv`
  (2 biến thể: giữ `-` / thay `-` bằng dấu cách; đã lowercase và tách `,` `.`).

## 1. Ví dụ thực tế

| Input `infer_sentence(...)` | Output |
|---|---|
| `a mít kế toán` | `a ˧˧ ▁ m i t ˦˥ ▁ k e ˨˦ ▁ t w a n ˨˦` |
| `a_mít kế_toán` | `a ˧˧ m i t ˦˥ ▁ k e ˨˦ t w a n ˨˦` |
| `a-mít kế toán` | `a ˧˧ - m i t ˦˥ ▁ k e ˨˦ ▁ t w a n ˨˦` |
| `a - mít` | `a ˧˧ ▁ - ▁ m i t ˦˥` |
| `tắc` | `t ă k ˦˥` |
| `xa` | `s a ˧˧` |
| `salagov` | `s a l a ɣ ɔ v ˧˧`  ← 3 âm tiết nhưng chỉ 1 thanh |
| `sa la gov` | `s a ˧˧ ▁ l a ˧˧ ▁ ɣ ɔ v ˧˧` |
| `mi-sa-sa-la-gov` | `m i ˧˧ - s a ˧˧ - s a ˧˧ - l a ˧˧ - ɣ ɔ v ˧˧` |
| `hỗ trợ` | `h o ˧ˀ˥ ▁ c ɤ ˨ˀ˩ ʔ` |
| `kế toán, phần mềm` | `k e ˨˦ ▁ t w a n ˨˦ ▁ f ɤ̆ n ˧˨ ▁ ...`  ← **dấu `,` bị mất** |
| `kế toán , phần mềm` | `k e ˨˦ ▁ t w a n ˨˦ ▁ , ▁ f ɤ̆ n ˧˨ ▁ ...` |
| `1` | `ɛ ˧˧`  ← sai, phải chuẩn hoá số trước |

## 2. Cấu trúc chuỗi

- Output là **một chuỗi token cách nhau bằng 1 dấu cách**. XPhoneBERT tokenizer là `WordLevel` + `WhitespaceSplit`,
  nên mỗi token cách nhau bởi space = 1 id. Không có normalizer.
- **Ranh giới từ = token `▁`** (U+2581, id 4), nối bằng `" ▁ "`. Một "từ" = một token tách theo dấu cách ở input,
  KHÔNG phải một âm tiết.
- **`_` ở input** (tham số `seperate_syllabel_token="_"`): được đổi thành dấu cách rồi tra từ điển hoặc chạy model.
  Các âm tiết trong một từ ghép nằm liền nhau, **không có ký hiệu ranh giới âm tiết** (`k e ˨˦ t w a n ˨˦`).
  Ranh giới âm tiết chỉ suy ra được từ vị trí token thanh.
- **`-` ở input**: không tách từ. Model sinh token `-` (id 93) giữa các âm tiết, nằm trong cùng một từ (không có `▁`).
  Có dạng này vì từ điển có các entry như `a-la → a˧˧-la˧˧`. Nếu thay `-` bằng dấu cách, mỗi âm tiết thành một từ có `▁`.
- **Dấu câu**: chỉ nhận ra khi là token riêng, tức có dấu cách hai bên (`. ? ! , : ; - ( ) ...`). Khi đó output
  là chính ký tự đó, kèm `▁` hai bên. Dấu câu dính vào chữ (`toán,`) thì đi vào model và **bị nuốt mất**.
- Mỗi âm vị là một token: phụ âm, nguyên âm, bán âm `w`/`j`, rồi **token thanh điệu ở cuối âm tiết**.
  Âm vị đa ký tự là một token: `tʰ`, `k͡p`, `ŋ͡m`, `ɤ̆`, `ă`.
- Unicode: `segments` trả về dạng **NFD**. `ă` trong output là `a`+U+0306, khớp vocab; dạng precomposed U+0103 **không có** trong vocab.
  Nếu tự ghép âm vị (không qua `segments`) phải `unicodedata.normalize("NFD", ...)`.

## 3. Thanh điệu (vie-n, chữ số thanh Chao, đặt sau phần vần)

| Thanh | Âm tiết mở / tận cùng m n ng nh | Âm tiết tắc (tận cùng p t c ch) |
|---|---|---|
| ngang | `˧˧` | `˧˧` (hiếm, chủ yếu từ ngoại lai: `net → n ɛ t ˧˧`) |
| huyền | `˧˨` | — |
| sắc | `˨˦` (`kế`, `toán`) | `˦˥` (`tắc`, `mít`) |
| hỏi | `˧˩˨` | — |
| ngã | `˧ˀ˥` | — |
| nặng | `˨ˀ˩` **+ token riêng `ʔ`** (`trợ → c ɤ ˨ˀ˩ ʔ`) | `˨ˀ˩` (không có `ʔ`) (`vực → v ɯ k ˨ˀ˩`) |

- Thanh sắc có 2 ký hiệu tuỳ loại âm tiết. Bộ kiểm tra không được giả định là 1 thanh ứng với 1 ký hiệu.
- Nặng ở âm tiết mở/vang thành **2 token** (`˨ˀ˩` `ʔ`). Khi đếm "1 âm tiết = 1 token thanh", coi `ʔ` là phụ.
- Token thanh hợp lệ: `˧˧ ˧˨ ˨˦ ˦˥ ˧˩˨ ˧ˀ˥ ˨ˀ˩` (và `ʔ` đi kèm nặng). Từ ngoại lai đôi khi sinh `˧` đơn
  (`jetpay → z ɛ t ˧ p ă j ˧˧`). Đây là dấu hiệu output lỗi.

## 4. G2P xử lý từ lạ ra sao

Luồng của `infer_sentence`, theo từng token tách bằng dấu cách:
1. Đổi `_` thành dấu cách.
2. Tra `phone_dict` theo **key khớp chính xác, phân biệt hoa thường, không chuẩn hoá Unicode**.
   (`infer_dataset` có `.lower()`, `infer_sentence` thì **không**.)
3. Nếu không có trong từ điển và là dấu câu: giữ nguyên.
4. Nếu không: sinh bằng ByT5 (prompt `<vie-n>: word`, greedy, `max_length=174`). **Không cache** giữa các câu.

Hành vi quan sát được với từ lạ / tiếng Anh:
- Từ VN thiếu trong từ điển (`hỗ`, `khuyến`, `nhiên`, `quảng`, `Lĩnh` viết hoa): model đoán đúng.
- Tên riêng phiên âm kiểu VN (`hắp`, `cừn`, `ây`, `mis`, `gov`): ra 1 âm tiết hợp lý, thanh `˧˧` nếu không dấu.
- **Từ nhiều âm tiết viết liền (`salagov`, `misa`): gộp thành 1 cụm, chỉ 1 thanh ở cuối.** Cấu trúc âm tiết bị mất.
- Tiếng Anh (`marketing`, `automation`, `invoice`...): ra kết quả không ổn định, có stress `ˈi`/`ˈa`,
  thanh lẻ `˧`, `marketing` ra 2 kiểu khác nhau tuỳ ngữ cảnh. Trên dataset có 12 từ như vậy, số thanh không khớp số âm tiết.
- Số: `1 → ɛ ˧˧` (sai). Cần chuẩn hoá số thành chữ trước.

Trên toàn bộ 226 câu: 13.266 token, 53 loại, **0 token OOV** với vocab XPhoneBERT. Lỗi nằm ở nội dung
(âm vị, thanh sai), không phải ở tokenizer.

## 5. Từ điển vie-n nằm ở đâu

- **Không đóng gói trong thư viện.** `__init__` chạy `os.system("wget https://raw.githubusercontent.com/lingjzhu/CharsiuG2P/main/dicts/vie-n.tsv")`
  vào **thư mục hiện tại (CWD)**, rồi đọc `./vie-n.tsv`. Trên Kaggle là `/kaggle/working/vie-n.tsv`.
- Trên Windows/máy không có `wget`, việc tải **thất bại im lặng**: dict rỗng và mọi từ đều đi qua model. Nên đặt sẵn file vào CWD.
- Format: `từ<TAB>phiên_âm`, UTF-8 NFC, 70.901 dòng, toàn chữ thường, không có entry nhiều cách đọc (không có dấu `,`).
  Phiên âm viết liền không cách (`kế toán → ke˨˦twan˨˦`). Một số key có `-` (`a-la`). Không có `_`.
- Phân bố: 7.502 key 1 âm tiết, 48.468 key 2 âm tiết, phần còn lại 3 đến 8+ âm tiết.

Lưu ý khi dùng làm bộ kiểm tra âm tiết:
- **Không đầy đủ.** Thiếu các âm tiết phổ biến: `hỗ`, `khuyến`, `nhiên`, `quảng`.
  Trong dataset có 49/371 âm tiết không có trong từ điển (gồm cả tên riêng/tiếng Anh).
- **Vị trí dấu thanh không nhất quán**: có `hoá` nhưng không có `hóa`. `thuỷ` và `thủy` đều có.
  Phải chuẩn hoá vị trí dấu (cả kiểu cũ lẫn mới) trước khi tra.
- Vì vậy bộ kiểm tra nên **kiểm tra theo luật chính tả** (âm đầu + vần + thanh, ràng buộc thanh sắc/nặng
  với âm tắc p/t/c/ch), và dùng `vie-n.tsv` để tham khảo/lấy phiên âm, không dùng nó làm whitelist duy nhất.

## 6. Pipeline train hiện có

- Trong `tts-amis/` chỉ có dataset (`metadata.csv` dạng `N.wav|text`, 226 dòng). Trên Kaggle account
  `triuinhhi` cũng không có notebook/dataset TTS nào. **Không có pipeline train nào để đối chiếu.**
- Dataset **không dùng tách từ**: không có `_`, có 283 dấu `-` để nối âm tiết của tên riêng (`a-mít`, `mi-sa-sa-la-gov`),
  có chữ hoa đầu câu, dấu `,` `.` dính chữ, và 1 chữ số (`…ích-1`).
- Upstream XPhoneBERT: input phải **tách từ bằng VnCoreNLP** (`_` nối âm tiết) và chuẩn hoá bằng Vinorm.
  Pretrain dùng cách này, nên chuỗi pretrain có `▁` giữa các *từ*, các âm tiết trong từ nằm liền nhau.

## 7. Format chốt cho các bước sau (đề xuất)

Tiền xử lý trước khi gọi `infer_sentence`:
1. `NFC`, rồi chuẩn hoá vị trí dấu thanh, rồi `lower()`.
2. Chuẩn hoá số và ký hiệu thành chữ (`1 → một`).
3. Tách dấu câu thành token riêng: `re.sub(r"\s*([,.!?;:])\s*", r" \1 ", t)`.
4. Tên riêng/tiếng Anh: viết lại thành các âm tiết VN hợp lệ, nối bằng `-` (`salagov → sa-la-gov`).
   Không để model tự đoán từ viết liền.
5. Ranh giới từ: chọn một trong hai cách và giữ nhất quán giữa train và infer.
   - (a) Giữ như dataset hiện tại (không tách từ, `-` trong tên riêng), hoặc
   - (b) VnCoreNLP `_` cho khớp phân phối pretrain của XPhoneBERT.

Output mong đợi, một dòng filelist:
`N.wav|a ˧˧ - m i t ˦˥ ▁ k e ˨˦ ▁ t w a n ˨˦ ▁ , ▁ ...`

Bất biến để kiểm tra tự động:
- Mọi token nằm trong vocab XPhoneBERT.
- Mỗi âm tiết có đúng 1 token thanh trong `{˧˧ ˧˨ ˨˦ ˦˥ ˧˩˨ ˧ˀ˥ ˨ˀ˩}`.
- `ʔ` chỉ đứng ngay sau `˨ˀ˩`.
- Không có `ˈ` `ˌ` `ː`.
- Không có `˧` đơn.
