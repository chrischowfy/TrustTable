import pandas as pd

def parse_structured_table(table_dict: dict) -> pd.DataFrame:
    """
    直接从结构化字典加载表格，100% 避免解析错误。
    """
    try:
        # 1. 提取表头和行
        header = table_dict.get("header", [])
        rows = table_dict.get("rows", [])

        # PubHealthTab uses matrix_data instead of header+rows
        if not header and not rows:
            matrix = table_dict.get("matrix_data", [])
            if matrix and len(matrix) >= 2:
                header = [str(c) for c in matrix[0]]
                rows = matrix[1:]
        
        # 2. 直接转换为 DataFrame
        df = pd.DataFrame(rows, columns=header)
        
        # 3. 清洗：去除换行符 \n 干扰（例如 'LOA\n(Metres)' -> 'LOA (Metres)'）
        # 处理列名
        df.columns = [c.replace('\n', ' ').strip() for c in df.columns]
        
        # 处理单元格内容（例如 'Ed Psaltis\nBob Thomas'）
        df = df.apply(lambda x: x.astype(str).str.replace('\n', ' ').str.strip())
        
        return df
    except Exception as e:
        print(f"Structured Table Load Error: {e}")
        return pd.DataFrame()