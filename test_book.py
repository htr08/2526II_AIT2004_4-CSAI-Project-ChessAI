import chess
import chess.polyglot

board = chess.Board()
book_path = "data/opening_books/performance.bin"

print(f"Đang kiểm tra file sách tại đường dẫn: {book_path}...")

try:
    with chess.polyglot.open_reader(book_path) as b:
        entries = list(b.find_all(board))
        print(f"✅ Thành công! Tìm thấy {len(entries)} nước cờ khai cuộc gợi ý cho thế trận ban đầu.")
        print("Gợi ý 3 nước đi hàng đầu từ sách:")
        for e in entries[:3]: 
            print(f"  -> Nước đi: {e.move} | Trọng số ưu tiên (weight) = {e.weight}")
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file ở đường dẫn '{book_path}'. Bạn hãy kiểm tra lại vị trí lưu file!")
except Exception as e:
    print(f"❌ Lỗi hệ thống: {e}")