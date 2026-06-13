# <Chưa biết đặt tiêu đề là gì>
## 1, Cài đặt môi trường
1. Chạy chuỗi lênh sau (khi mới clone về máy):  
    ```powershell
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
    . .\pipeline.ps1
    ```
2. Thao tác sau khi set up gồm:
    ```powershell
    # Check + activate môi trường ảo
    . .\pipeline.ps1

    # Deactivate môi trường ảo
    deactivate
    ```

## 2, Cài đặt mô hình của mình:
1. Cấu trúc lưu mô hình như sau:  
```text
root/
| ...
| Model_dir/
    | Your_folder/
        | architecture.py
        | raw_metadata.json
```

2. Cần kiểm tra:  
    - Cấu trúc ***architecture.py*** (Dựa vào Khánh's folder)
    - Đường **link drive** trong ***raw_metadata.json***

3. Sau khi đã chắc chắn tất cả thông tin trên, chạy lệnh sau:
    ```powershell
    python refactor_checkpoints.py --root Model_dir --json-log
    ```
    > 1. Lệnh này sẽ đọc file .pt từ đường dẫn drive và tái cấu trúc lại (chỉ còn lại tham số) rồi lưu ngược về repo.
    > 2. Tất cả các file .pt và các kết quả sinh ra sau lệnh này sẽ không được push lên repo (trong .gitignore).

4. **Lưu ý:** Đọc kĩ log trước khi sang bước tiếp theo.
## 3, Coming soon:
> Hãy clone về và chạy theo pipeline này để test tiến độ, nếu có vấn đề, hãy báo ngay cho tôi