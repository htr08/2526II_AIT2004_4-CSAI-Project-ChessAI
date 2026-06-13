# <Hiện đang cập nhật Pipeline chạy tournament của dự án.>
- Branch đang cập nhật pipeline, hiện tại chỉ có cài đặt môi trường ảo:  
- Trên terminal chạy lệnh theo hướng dẫn sau:

```powershell
# Set quyền chạy powershell cho file git clone
# (Chạy 1 lần duy nhất thôi)
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# Chạy chuỗi check + install(nếu k có env) + active env
. .\pipeline.ps1

# Nếu muốn thoát môi trường .venv
deactivate
```