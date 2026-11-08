# BUSI 勘誤清單

`dataset_comment_list.csv` 原檔取自 Pawłowska、Karwat、Żołek（2023），
[Supplementary files DIB-D-23-00350, v1](https://doi.org/10.17632/k8t3gnx9h6.1)，授權 CC BY 4.0。

- [原始檔下載](https://data.mendeley.com/public-files/datasets/k8t3gnx9h6/files/41d7e047-07b3-4a4a-b89f-b9b70f6e9285/file_downloaded)
- SHA-256：`e3c2abc0ce027401dc717bf8a7ec76442e30c3cda2df32a0bc8aa197ff1c74a4`
- 保留原始內容及分號分隔格式，未修改作者分組。
- 使用作者 `Filename` 欄的群組，涵蓋經幾何變換、近似或同一病灶序列的重複影像；不是只比對檔名。
- 群組第一張為預先指定代表，`Objection` 非空者整組排除；不依模型輸出、遮罩形狀或測試分數挑選。
- `Annotation` 僅保留紀錄，這次不因文字、量測或都卜勒標記排除。
- 程式另外合併像素完全相同的影像群組；來源影像不刪除或改寫。

引用：Pawłowska A., Karwat P., Żołek N. Letter to the Editor. Data in Brief 48 (2023), 109247.
[勘誤論文](https://doi.org/10.1016/j.dib.2023.109247)
