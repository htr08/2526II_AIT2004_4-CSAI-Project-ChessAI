"""Hiển thị bàn cờ màu sắc trên terminal để kiểm tra trạng thái game."""

import chess

def print_board_fancy(board):
    """
    In bàn cờ ra Terminal một cách trực quan, có màu sắc (ANSI) để dễ debug.
    Quân viết HOA là Trắng, viết thường là Đen.
    """
    # Mã màu ANSI cơ bản để hiển thị đẹp hơn trên Terminal
    RESET = "\033[0m"
    BOLD = "\033[1m"
    WHITE_COLOR = "\033[97m"  # Trắng sáng
    BLACK_COLOR = "\033[90m"  # Đen xám (hoặc dùng màu khác tùy terminal)
    
    print("\n   a b c d e f g h")
    print(" +-----------------+")
    
    for rank in range(7, -1, -1):
        row = f"{rank+1}| "
        for file in range(8):
            sq = chess.square(file, rank)
            p = board.piece_at(sq)
            
            if p:
                symbol = p.symbol()
                # Nếu là quân Trắng (viết HOA), tô màu sáng + đậm
                if p.color == chess.WHITE:
                    row += f"{BOLD}{WHITE_COLOR}{symbol}{RESET} "
                else:
                    # Nếu là quân Đen (viết thường), tô màu tối hơn
                    row += f"{BLACK_COLOR}{symbol}{RESET} "
            else:
                row += ". "
                
        row += f"| {rank+1}"
        print(row)
        
    print(" +-----------------+")
    print("   a b c d e f g h\n")

# Khối này chỉ chạy khi thực thi trực tiếp file này (python scripts/explore_board.py)
if __name__ == "__main__":
    print("--- Thử nghiệm hiển thị bàn cờ ảo ---")
    board = chess.Board()
    
    # Đi thử vài nước khai cuộc
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    board.push_uci("g1f3")
    board.push_uci("b8c6")
    
    print_board_fancy(board)
    print("FEN hiện tại:", board.fen())