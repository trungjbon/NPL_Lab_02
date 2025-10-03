# 1. Các bước triển khai

## Khởi tạo Spark Session

- Dùng SparkSession.builder với chế độ local[*] để chạy Spark trên máy cá nhân.

## Đọc dữ liệu C4 dataset

- File dữ liệu ở dạng JSON nén .gz.

- Sử dụng spark.read.json() để load dữ liệu vào DataFrame.

- Giới hạn limit(1000) bản ghi để xử lý nhanh hơn trong quá trình thử nghiệm.

- Đo thời gian đọc dữ liệu và log lại.

## Xây dựng Spark ML Pipeline gồm các stage:

- RegexTokenizer: tách văn bản thành tokens dựa trên khoảng trắng và dấu câu.

- StopWordsRemover: loại bỏ stopwords.

- HashingTF: chuyển tokens thành vector tần suất.

- IDF: tính toán trọng số Inverse Document Frequency và tạo TF-IDF vector.

- Normalizer: chuẩn hóa vector TF-IDF về chuẩn L2 để có thể tính cosine similarity qua dot product.

## Huấn luyện và biến đổi dữ liệu

- Dùng pipeline.fit(initialDF) để train pipeline.

- Dùng pipelineModel.transform(initialDF) để áp dụng pipeline cho dữ liệu gốc.

## Tìm k văn bản giống nhau

- Tính vector chuẩn hóa cho document query.

- Với toàn bộ tập dữ liệu, tính cosine similarity.

- Sắp xếp giảm dần và in ra Top-K similar documents đầu tiên.

## Lưu kết quả và log

- Kết quả TF-IDF vectors được lưu vào file ../results/lab17_pipeline_output.txt.

- Metrics được lưu vào file log ../log/lab17_metrics.log, bao gồm:
  - Thời gian read dataset

  - Thời gian fit pipeline

  - Thời gian transform dữ liệu

  - Vocabulary size
    
  - HashingTF Num Features

# 2. Cách chạy chương trình

## Yêu cầu cài đặt:

- JDK

- SBT để build project

## Chạy ứng dụng:
- Ví dụ:
  - 10: số documents lấy ra từ dataset.
  - 0: chỉ số document được chọn để tìm similar docs.
  - 5: top k similar docs.
```
sbt "run 10 0 5"
```
- Các biến đầu vào có thể thay đổi.

# 3. Giải thích kết quả

- Schema dữ liệu: có cột text chứa văn bản.

- Tokenization: văn bản được tách thành từ riêng lẻ. Ví dụ "Hello world!" → ["Hello", "world"].

- StopWordsRemover: loại bỏ các từ dừng phổ biến như "the", "and", "is".

- TF-IDF vectorization:

  - HashingTF ánh xạ từ → index trong vector 20,000 chiều.

  - IDF tính toán độ quan trọng của từ trong toàn bộ corpus.

  - Kết quả: mỗi văn bản được biểu diễn dưới dạng sparse vector với TF-IDF values.

- Normalizer: chuẩn hóa vector về độ dài 1 (chuẩn L2).

- Find Similar Docs: Tính cosine similarity và in ra Top-K docs tương tự, kèm similarity score.

- Ví dụ một phần file output.txt:
```
Original Text: I thought I was going to finish the 3rd season of the Wire tonight.
But there was a commentary on ep...
TF-IDF Vector: (20000,[749,3389,3401,3491,5038,6487,6578,7412,8273,10055,12309,12358,17464,18431,18458,19878],[5.116995310087166,4.51085950651685,10.59863373376224,5.29931686688112,2.0489423749535485,3.1245651453969594,6.949535149660148,4.200704578213011,2.1465808445174646,2.781620394270129,3.730700948967275,6.204184579089802,3.96431580014878,3.147554663621658,3.443018876515494,3.5075573976530654])
```
- Điều này cho thấy mỗi document được biểu diễn bằng vector số học.
  
- Ví dụ output phần similar documents:
```
[info] Top 5 similar documents to doc #0:
[info] +--------------------+--------------------+
[info] |                text|          similarity|
[info] +--------------------+--------------------+
[info] |Beginners BBQ Cla...|                 1.0|
[info] |The rich get rich...|0.015918218766508528|
[info] |How many backlink...|0.010374121156947374|
[info] |Biomedics 1 Day E...| 0.00963145074612207|
[info] |Discussion in 'Ma...|0.008126424202131027|
[info] +--------------------+--------------------+
```


# 4. Khó khăn và giải pháp

# Phần log terminal gây rối
- Lược bỏ bớt info bớt quan trọng.
- Giải pháp:
```
import org.apache.log4j.{Level, Logger}
Logger.getRootLogger.setLevel(Level.WARN)
```



